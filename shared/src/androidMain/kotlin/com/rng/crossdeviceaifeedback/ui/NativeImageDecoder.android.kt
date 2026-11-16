package com.rng.crossdeviceaifeedback.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory

actual fun decodeImageToNormalizedFloatArray(
    imageBytes: ByteArray,
    targetWidth: Int,
    targetHeight: Int
): FloatArray {
    if (imageBytes.isEmpty()) return FloatArray(3 * targetWidth * targetHeight) { 0.0f }

    val originalBitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        ?: return FloatArray(3 * targetWidth * targetHeight) { 0.0f }

    // Center crop to 1:1 square ratio to prevent object stretching/distortion
    val srcWidth = originalBitmap.width
    val srcHeight = originalBitmap.height
    val minDim = minOf(srcWidth, srcHeight)
    val cropX = (srcWidth - minDim) / 2
    val cropY = (srcHeight - minDim) / 2

    val croppedBitmap = Bitmap.createBitmap(originalBitmap, cropX, cropY, minDim, minDim)
    val scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, targetWidth, targetHeight, true)

    val pixels = IntArray(targetWidth * targetHeight)
    scaledBitmap.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)

    val floatArray = FloatArray(3 * targetWidth * targetHeight)

    var idx = 0
    val numPixels = targetWidth * targetHeight
    for (i in 0 until numPixels) {
        val color = pixels[i]
        val r = ((color shr 16) and 0xFF)
        val g = ((color shr 8) and 0xFF)
        val b = (color and 0xFF)

        // MobileNet NHWC format [R, G, B, R, G, B...] normalized to [-1.0, 1.0]
        floatArray[idx++] = (r - 127.5f) / 127.5f
        floatArray[idx++] = (g - 127.5f) / 127.5f
        floatArray[idx++] = (b - 127.5f) / 127.5f
    }
    return floatArray
}
