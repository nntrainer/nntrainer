package com.example.quickdotai

import kotlin.math.abs
import kotlin.math.ceil

/**
 * Pillow's `ImagingResample` in Kotlin: a separable two-pass resample with
 * support scaling (i.e. antialiasing on downscale) and 22-bit fixed-point
 * weights, so the output matches `PIL.Image.resize()` to within a LSB.
 *
 * The filter is a (kernel, support) pair, exactly as Pillow models it:
 *   BILINEAR -> support 1.0,  BICUBIC -> support 2.0 (Catmull-Rom a = -0.5).
 *
 * [PillowBicubicResizer] implements `resample: 3` and
 * [PillowBilinearResizer] implements `resample: 2` from processor configs.
 */
internal object PillowResample {

    /** A resample filter: the kernel and its (unscaled) support radius. */
    internal class Filter(val support: Double, val kernel: (Double) -> Double)

    val BILINEAR = Filter(1.0) { x ->
        val absX = abs(x)
        if (absX < 1.0) 1.0 - absX else 0.0
    }

    /**
     * Pillow's BICUBIC: the Keys cubic with a = -0.5 (Catmull-Rom), the same
     * kernel torchvision/PIL use for `resample=3`.
     */
    val BICUBIC = Filter(2.0) { x ->
        val a = -0.5
        val absX = abs(x)
        when {
            absX < 1.0 -> ((a + 2.0) * absX - (a + 3.0)) * absX * absX + 1.0
            absX < 2.0 -> (((absX - 5.0) * absX + 8.0) * absX - 4.0) * a
            else -> 0.0
        }
    }

    private const val PRECISION_BITS = 22

    /** Precomputed per-output-pixel bounds + fixed-point weights for one axis. */
    private class Coeffs(val bounds: IntArray, val kk: IntArray, val kSize: Int)

    private fun computeCoeffs(inSize: Int, outSize: Int, filter: Filter): Coeffs {
        val scale = inSize.toDouble() / outSize
        val filterScale = if (scale < 1.0) 1.0 else scale
        val support = filter.support * filterScale
        val invFilterScale = 1.0 / filterScale

        val kSize = ceil(support).toInt() * 2 + 1
        val bounds = IntArray(outSize * 2)
        val kk = IntArray(outSize * kSize)

        for (o in 0 until outSize) {
            val center = (o + 0.5) * scale
            var min = (center - support + 0.5).toInt()
            var max = (center + support + 0.5).toInt()
            if (min < 0) min = 0
            if (max > inSize) max = inSize
            val count = max - min

            bounds[o * 2] = min
            bounds[o * 2 + 1] = count

            var ww = 0.0
            val weights = DoubleArray(count)
            for (i in 0 until count) {
                val w = filter.kernel((min + i + 0.5 - center) * invFilterScale)
                weights[i] = w
                ww += w
            }
            for (i in 0 until count) {
                if (ww != 0.0) weights[i] /= ww
                kk[o * kSize + i] = if (weights[i] < 0) {
                    (weights[i] * (1 shl PRECISION_BITS) - 0.5).toInt()
                } else {
                    (weights[i] * (1 shl PRECISION_BITS) + 0.5).toInt()
                }
            }
        }
        return Coeffs(bounds, kk, kSize)
    }

    fun resize(
        pixels: IntArray,
        width: Int,
        height: Int,
        newWidth: Int,
        newHeight: Int,
        filter: Filter
    ): IntArray {
        val halfOne = 1 shl (PRECISION_BITS - 1)

        // 1. Horizontal pass: (width, height) -> (newWidth, height)
        val cx = computeCoeffs(width, newWidth, filter)
        val tempPixels = IntArray(newWidth * height)
        for (y in 0 until height) {
            for (x in 0 until newWidth) {
                val xMin = cx.bounds[x * 2]
                val count = cx.bounds[x * 2 + 1]
                var r = halfOne
                var g = halfOne
                var b = halfOne
                var a = halfOne
                for (i in 0 until count) {
                    val weight = cx.kk[x * cx.kSize + i]
                    val pixel = pixels[y * width + xMin + i]
                    a += ((pixel shr 24) and 0xFF) * weight
                    r += ((pixel shr 16) and 0xFF) * weight
                    g += ((pixel shr 8) and 0xFF) * weight
                    b += (pixel and 0xFF) * weight
                }
                tempPixels[y * newWidth + x] =
                    (((a shr PRECISION_BITS).coerceIn(0, 255)) shl 24) or
                        (((r shr PRECISION_BITS).coerceIn(0, 255)) shl 16) or
                        (((g shr PRECISION_BITS).coerceIn(0, 255)) shl 8) or
                        ((b shr PRECISION_BITS).coerceIn(0, 255))
            }
        }

        // 2. Vertical pass: (newWidth, height) -> (newWidth, newHeight)
        val cy = computeCoeffs(height, newHeight, filter)
        val finalPixels = IntArray(newWidth * newHeight)
        for (x in 0 until newWidth) {
            for (y in 0 until newHeight) {
                val yMin = cy.bounds[y * 2]
                val count = cy.bounds[y * 2 + 1]
                var r = halfOne
                var g = halfOne
                var b = halfOne
                var a = halfOne
                for (i in 0 until count) {
                    val weight = cy.kk[y * cy.kSize + i]
                    val pixel = tempPixels[(yMin + i) * newWidth + x]
                    a += ((pixel shr 24) and 0xFF) * weight
                    r += ((pixel shr 16) and 0xFF) * weight
                    g += ((pixel shr 8) and 0xFF) * weight
                    b += (pixel and 0xFF) * weight
                }
                finalPixels[y * newWidth + x] =
                    (((a shr PRECISION_BITS).coerceIn(0, 255)) shl 24) or
                        (((r shr PRECISION_BITS).coerceIn(0, 255)) shl 16) or
                        (((g shr PRECISION_BITS).coerceIn(0, 255)) shl 8) or
                        ((b shr PRECISION_BITS).coerceIn(0, 255))
            }
        }
        return finalPixels
    }
}

/** `PIL.Image.resize(..., Image.BILINEAR)`. */
class PillowBilinearResizer {
    companion object {
        fun resize(
            pixels: IntArray,
            width: Int,
            height: Int,
            newWidth: Int,
            newHeight: Int
        ): IntArray = PillowResample.resize(
            pixels, width, height, newWidth, newHeight, PillowResample.BILINEAR
        )
    }
}

/**
 * `PIL.Image.resize(..., Image.BICUBIC)` (`processor_config.json` `"resample": 3`).
 */
class PillowBicubicResizer {
    companion object {
        fun resize(
            pixels: IntArray,
            width: Int,
            height: Int,
            newWidth: Int,
            newHeight: Int
        ): IntArray = PillowResample.resize(
            pixels, width, height, newWidth, newHeight, PillowResample.BICUBIC
        )
    }
}
