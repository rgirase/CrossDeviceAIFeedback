package com.rng.crossdeviceaifeedback.export

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import kotlinx.cinterop.BetaInteropApi
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import platform.Foundation.NSData
import platform.Foundation.NSTemporaryDirectory
import platform.Foundation.NSURL
import platform.Foundation.create
import platform.Foundation.writeToFile
import platform.UIKit.UIActivityViewController
import platform.UIKit.UIApplication
import platform.UIKit.UIWindow

@OptIn(ExperimentalForeignApi::class, BetaInteropApi::class)
@Composable
actual fun rememberFileSharer(): FileSharer {
    return remember {
        object : FileSharer {
            override fun shareFile(filename: String, content: String) {
                try {
                    val bytes = content.encodeToByteArray()
                    val nsData = bytes.usePinned { pinned ->
                        NSData.create(bytes = pinned.addressOf(0), length = bytes.size.toULong())
                    }

                    val tempDir = NSTemporaryDirectory()
                    val filePath = "$tempDir/$filename"
                    val fileUrl = NSURL.fileURLWithPath(filePath)

                    nsData.writeToFile(filePath, true)

                    val activityItems = listOf(fileUrl)
                    val activityVc = UIActivityViewController(activityItems = activityItems, applicationActivities = null)

                    val rootVc = UIApplication.sharedApplication.keyWindow?.rootViewController
                        ?: (UIApplication.sharedApplication.windows.firstOrNull() as? UIWindow)?.rootViewController

                    rootVc?.presentViewController(activityVc, animated = true, completion = null)

                } catch (e: Exception) {
                    println("iOS Share Error: ${e.message}")
                }
            }
        }
    }
}
