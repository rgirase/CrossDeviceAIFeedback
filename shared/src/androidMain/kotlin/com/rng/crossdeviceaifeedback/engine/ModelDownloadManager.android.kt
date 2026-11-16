package com.rng.crossdeviceaifeedback.engine

import java.io.File

actual fun checkLocalModelExists(filename: String): Boolean {
    val ctx = androidApplicationContext
    if (ctx != null) {
        val file = File(ctx.filesDir, filename)
        if (file.exists() && file.length() > 0) return true

        try {
            val fileDescriptor = ctx.assets.openFd(filename)
            if (fileDescriptor.declaredLength > 0) return true
        } catch (_: Exception) {
        }
    }
    return false
}

actual fun saveDownloadedModelFile(filename: String, bytes: ByteArray): String? {
    val ctx = androidApplicationContext ?: return null
    return try {
        val file = File(ctx.filesDir, filename)
        file.writeBytes(bytes)
        file.absolutePath
    } catch (_: Exception) {
        null
    }
}
