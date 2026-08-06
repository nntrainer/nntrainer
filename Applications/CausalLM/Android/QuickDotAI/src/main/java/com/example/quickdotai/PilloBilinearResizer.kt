package com.example.quickdotai

import kotlin.math.abs
import kotlin.math.ceil

class PillowBilinearResizer {
    companion object {

        // Bilinear kernel
        private fun bilinearKernel(x: Double): Double {
            val absX = abs(x)
            if (absX < 1.0) {
                return 1.0 - absX
            }
            return 0.0
        }

        fun resize(
            pixels: IntArray,
            width: Int,
            height: Int,
            newWidth: Int,
            newHeight: Int
        ): IntArray {
            // Pillow uses fixed point arithmetic with 22 bits of precision for weights
            val precisionBits = 22
            val halfOne = 1 shl (precisionBits - 1)

            // 1. Horizontal Pass: (width, height) -> (newWidth, height)
            val tempPixels = IntArray(newWidth * height)
            val xScale = width.toDouble() / newWidth
            val filterScaleX = if (xScale < 1.0) 1.0 else xScale
            val supportX = 1.0 * filterScaleX // Support is 1.0 for Bilinear
            val scaleFactorX = 1.0 / filterScaleX

            // Precompute weights for horizontal pass
            val kSizeX = (ceil(supportX).toInt() * 2 + 1)
            val boundsX = IntArray(newWidth * 2)
            val kkX = IntArray(newWidth * kSizeX)

            for (x in 0 until newWidth) {
                val center = (x + 0.5) * xScale
                var xMin = (center - supportX + 0.5).toInt()
                var xMax = (center + supportX + 0.5).toInt()

                if (xMin < 0) xMin = 0
                if (xMax > width) xMax = width

                val count = xMax - xMin

                boundsX[x * 2] = xMin
                boundsX[x * 2 + 1] = count

                var ww = 0.0
                val weights = DoubleArray(count)
                for (i in 0 until count) {
                    val srcX = xMin + i
                    val w = bilinearKernel((srcX + 0.5 - center) * scaleFactorX)
                    weights[i] = w
                    ww += w
                }

                // Normalize and convert to fixed point
                for (i in 0 until count) {
                    if (ww != 0.0) weights[i] /= ww
                    val fw = if (weights[i] < 0) {
                        (weights[i] * (1 shl precisionBits) - 0.5).toInt()
                    } else {
                        (weights[i] * (1 shl precisionBits) + 0.5).toInt()
                    }
                    kkX[x * kSizeX + i] = fw
                }
            }

            for (y in 0 until height) {
                for (x in 0 until newWidth) {
                    val xMin = boundsX[x * 2]
                    val count = boundsX[x * 2 + 1]

                    var r = halfOne
                    var g = halfOne
                    var b = halfOne
                    var a = halfOne

                    for (i in 0 until count) {
                        val weight = kkX[x * kSizeX + i]
                        val srcX = xMin + i
                        val pixel = pixels[y * width + srcX]

                        a += ((pixel shr 24) and 0xFF) * weight
                        r += ((pixel shr 16) and 0xFF) * weight
                        g += ((pixel shr 8) and 0xFF) * weight
                        b += (pixel and 0xFF) * weight
                    }

                    val rInt = (r shr precisionBits).coerceIn(0, 255)
                    val gInt = (g shr precisionBits).coerceIn(0, 255)
                    val bInt = (b shr precisionBits).coerceIn(0, 255)
                    val aInt = (a shr precisionBits).coerceIn(0, 255)

                    tempPixels[y * newWidth + x] =
                        (aInt shl 24) or (rInt shl 16) or (gInt shl 8) or bInt
                }
            }

            // 2. Vertical Pass: (newWidth, height) -> (newWidth, newHeight)
            val finalPixels = IntArray(newWidth * newHeight)
            val yScale = height.toDouble() / newHeight
            val filterScaleY = if (yScale < 1.0) 1.0 else yScale
            val supportY = 1.0 * filterScaleY // Support is 1.0 for Bilinear
            val scaleFactorY = 1.0 / filterScaleY

            val kSizeY = (ceil(supportY).toInt() * 2 + 1)
            val boundsY = IntArray(newHeight * 2)
            val kkY = IntArray(newHeight * kSizeY)

            for (y in 0 until newHeight) {
                val center = (y + 0.5) * yScale
                var yMin = (center - supportY + 0.5).toInt()
                var yMax = (center + supportY + 0.5).toInt()

                if (yMin < 0) yMin = 0
                if (yMax > height) yMax = height

                val count = yMax - yMin

                boundsY[y * 2] = yMin
                boundsY[y * 2 + 1] = count

                var ww = 0.0
                val weights = DoubleArray(count)
                for (i in 0 until count) {
                    val srcY = yMin + i
                    val w = bilinearKernel((srcY + 0.5 - center) * scaleFactorY)
                    weights[i] = w
                    ww += w
                }

                for (i in 0 until count) {
                    if (ww != 0.0) weights[i] /= ww
                    val fw = if (weights[i] < 0) {
                        (weights[i] * (1 shl precisionBits) - 0.5).toInt()
                    } else {
                        (weights[i] * (1 shl precisionBits) + 0.5).toInt()
                    }
                    kkY[y * kSizeY + i] = fw
                }
            }

            for (x in 0 until newWidth) {
                for (y in 0 until newHeight) {
                    val yMin = boundsY[y * 2]
                    val count = boundsY[y * 2 + 1]

                    var r = halfOne
                    var g = halfOne
                    var b = halfOne
                    var a = halfOne

                    for (i in 0 until count) {
                        val weight = kkY[y * kSizeY + i]
                        val srcY = yMin + i
                        val pixel = tempPixels[srcY * newWidth + x]

                        a += ((pixel shr 24) and 0xFF) * weight
                        r += ((pixel shr 16) and 0xFF) * weight
                        g += ((pixel shr 8) and 0xFF) * weight
                        b += (pixel and 0xFF) * weight
                    }

                    val rInt = (r shr precisionBits).coerceIn(0, 255)
                    val gInt = (g shr precisionBits).coerceIn(0, 255)
                    val bInt = (b shr precisionBits).coerceIn(0, 255)
                    val aInt = (a shr precisionBits).coerceIn(0, 255)

                    finalPixels[y * newWidth + x] =
                        (aInt shl 24) or (rInt shl 16) or (gInt shl 8) or bInt
                }
            }

            return finalPixels
        }

    }
}
