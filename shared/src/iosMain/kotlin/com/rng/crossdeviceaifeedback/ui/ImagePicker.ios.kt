package com.rng.crossdeviceaifeedback.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import platform.UIKit.UIApplication
import platform.UIKit.UIImage
import platform.UIKit.UIImageJPEGRepresentation
import platform.UIKit.UIImagePickerController
import platform.UIKit.UIImagePickerControllerDelegateProtocol
import platform.UIKit.UIImagePickerControllerSourceType
import platform.UIKit.UINavigationControllerDelegateProtocol
import platform.UIKit.UIWindow
import platform.darwin.NSObject

@OptIn(ExperimentalForeignApi::class)
@Composable
actual fun rememberImagePickerLauncher(onImagePicked: (ByteArray?, String?) -> Unit): ImagePickerLauncher {
    return remember(onImagePicked) {
        object : ImagePickerLauncher {
            override fun launch() {
                val picker = UIImagePickerController()
                picker.sourceType = UIImagePickerControllerSourceType.UIImagePickerControllerSourceTypePhotoLibrary

                val delegate = object : NSObject(), UIImagePickerControllerDelegateProtocol, UINavigationControllerDelegateProtocol {
                    override fun imagePickerController(
                        picker: UIImagePickerController,
                        didFinishPickingMediaWithInfo: Map<Any?, *>
                    ) {
                        picker.dismissViewControllerAnimated(true, completion = null)
                        val image = didFinishPickingMediaWithInfo["UIImagePickerControllerOriginalImage"] as? UIImage
                        if (image != null) {
                            val data = UIImageJPEGRepresentation(image, 0.85)
                            if (data != null) {
                                val bytes = ByteArray(data.length.toInt())
                                bytes.usePinned { pinned ->
                                    platform.posix.memcpy(pinned.addressOf(0), data.bytes, data.length)
                                }
                                onImagePicked(bytes, "iOS_Gallery_Photo.jpg")
                            } else {
                                onImagePicked(null, null)
                            }
                        } else {
                            onImagePicked(null, null)
                        }
                    }

                    override fun imagePickerControllerDidCancel(picker: UIImagePickerController) {
                        picker.dismissViewControllerAnimated(true, completion = null)
                        onImagePicked(null, null)
                    }
                }

                picker.delegate = delegate

                val rootVc = UIApplication.sharedApplication.keyWindow?.rootViewController
                    ?: (UIApplication.sharedApplication.windows.firstOrNull() as? UIWindow)?.rootViewController

                rootVc?.presentViewController(picker, animated = true, completion = null)
            }
        }
    }
}
