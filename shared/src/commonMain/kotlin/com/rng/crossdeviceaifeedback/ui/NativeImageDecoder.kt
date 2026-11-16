package com.rng.crossdeviceaifeedback.ui

expect fun decodeImageToNormalizedFloatArray(
    imageBytes: ByteArray,
    targetWidth: Int = 224,
    targetHeight: Int = 224
): FloatArray
