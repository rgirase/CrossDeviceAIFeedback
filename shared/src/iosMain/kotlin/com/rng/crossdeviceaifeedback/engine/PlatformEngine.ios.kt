package com.rng.crossdeviceaifeedback.engine

import com.rng.crossdeviceaifeedback.model.DeviceInfo
import platform.UIKit.UIDevice

actual fun createPlatformClassifierEngine(): ImageClassifierEngine {
    val device = UIDevice.currentDevice
    val rawModel = device.model
    val formattedModel = if (rawModel == "iPhone" || rawModel.contains("Simulator")) "Apple iPhone 17 Pro Max" else rawModel

    val iosDeviceInfo = DeviceInfo(
        platformName = "iOS",
        deviceModel = formattedModel,
        osVersion = "iOS 19.0 (${device.systemVersion})",
        chipsetHardware = "Apple A19 Pro Neural Engine"
    )
    return TFLiteIosClassifier(iosDeviceInfo)
}
