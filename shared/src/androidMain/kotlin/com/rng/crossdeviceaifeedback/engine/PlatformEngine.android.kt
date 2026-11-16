package com.rng.crossdeviceaifeedback.engine

import android.content.Context
import android.os.Build
import com.rng.crossdeviceaifeedback.model.DeviceInfo

var androidApplicationContext: Context? = null

actual fun createPlatformClassifierEngine(): ImageClassifierEngine {
    val rawHardware = Build.HARDWARE ?: "ARM Processor"
    val formattedHardware = if (rawHardware.contains("grizzly", ignoreCase = true) || rawHardware.contains("tensor", ignoreCase = true)) "Google Tensor G6 NPU" else rawHardware

    val androidDeviceInfo = DeviceInfo(
        platformName = "Android",
        deviceModel = "${Build.MANUFACTURER} ${Build.MODEL}",
        osVersion = "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})",
        chipsetHardware = formattedHardware
    )
    return TFLiteAndroidClassifier(androidDeviceInfo, androidApplicationContext)
}
