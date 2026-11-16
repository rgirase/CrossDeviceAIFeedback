package com.rng.crossdeviceaifeedback.engine

actual fun checkLocalModelExists(filename: String): Boolean {
    return true
}

actual fun saveDownloadedModelFile(filename: String, bytes: ByteArray): String? {
    return "/tmp/$filename"
}
