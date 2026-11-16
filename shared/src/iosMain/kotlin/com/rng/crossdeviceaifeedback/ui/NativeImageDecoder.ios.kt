package com.rng.crossdeviceaifeedback.ui

import kotlinx.cinterop.BetaInteropApi
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import platform.CoreGraphics.CGBitmapContextCreate
import platform.CoreGraphics.CGColorSpaceCreateDeviceRGB
import platform.CoreGraphics.CGContextDrawImage
import platform.CoreGraphics.CGImageAlphaInfo
import platform.CoreGraphics.CGRectMake
import platform.CoreGraphics.CGSizeMake
import platform.Foundation.NSData
import platform.Foundation.create
import platform.UIKit.UIGraphicsBeginImageContextWithOptions
import platform.UIKit.UIGraphicsEndImageContext
import platform.UIKit.UIGraphicsGetImageFromCurrentImageContext
import platform.UIKit.UIImage

@OptIn(ExperimentalForeignApi::class, BetaInteropApi::class)
actual fun decodeImageToNormalizedFloatArray(
    imageBytes: ByteArray,
    targetWidth: Int,
    targetHeight: Int
): FloatArray {
    val totalFloats = 3 * targetWidth * targetHeight
    if (imageBytes.isEmpty()) return FloatArray(totalFloats) { 0.0f }

    val nsData = imageBytes.usePinned { pinned ->
        NSData.create(bytes = pinned.addressOf(0), length = imageBytes.size.toULong())
    }

    val rawImage = UIImage.imageWithData(nsData) ?: return FloatArray(totalFloats) { 0.0f }

    // Render image with UIGraphics to automatically fix EXIF orientation (rotation) & aspect ratio
    UIGraphicsBeginImageContextWithOptions(
        size = CGSizeMake(targetWidth.toDouble(), targetHeight.toDouble()),
        opaque = false,
        scale = 1.0
    )
    rawImage.drawInRect(CGRectMake(0.0, 0.0, targetWidth.toDouble(), targetHeight.toDouble()))
    val normalizedImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()

    val cgImage = normalizedImage?.CGImage ?: return FloatArray(totalFloats) { 0.0f }

    val colorSpace = CGColorSpaceCreateDeviceRGB()
    val rawPixels = ByteArray(4 * targetWidth * targetHeight)

    val context = rawPixels.usePinned { pinned ->
        CGBitmapContextCreate(
            data = pinned.addressOf(0),
            width = targetWidth.toULong(),
            height = targetHeight.toULong(),
            bitsPerComponent = 8u,
            bytesPerRow = (4 * targetWidth).toULong(),
            space = colorSpace,
            bitmapInfo = CGImageAlphaInfo.kCGImageAlphaPremultipliedLast.value
        )
    } ?: return FloatArray(totalFloats) { 0.0f }

    CGContextDrawImage(
        context,
        CGRectMake(0.0, 0.0, targetWidth.toDouble(), targetHeight.toDouble()),
        cgImage
    )

    val floatArray = FloatArray(totalFloats)
    var idx = 0
    val numPixels = targetWidth * targetHeight
    for (i in 0 until numPixels) {
        val p = i * 4
        val r = rawPixels[p].toInt() and 0xFF
        val g = rawPixels[p + 1].toInt() and 0xFF
        val b = rawPixels[p + 2].toInt() and 0xFF

        // NHWC [R, G, B, R, G, B...] normalized to [-1.0, 1.0]
        floatArray[idx++] = (r - 127.5f) / 127.5f
        floatArray[idx++] = (g - 127.5f) / 127.5f
        floatArray[idx++] = (b - 127.5f) / 127.5f
    }

    return floatArray
}
