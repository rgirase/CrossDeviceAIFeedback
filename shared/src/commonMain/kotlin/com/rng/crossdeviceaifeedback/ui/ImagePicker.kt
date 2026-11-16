package com.rng.crossdeviceaifeedback.ui

import androidx.compose.runtime.Composable

interface ImagePickerLauncher {
    fun launch()
}

@Composable
expect fun rememberImagePickerLauncher(onImagePicked: (ByteArray?, String?) -> Unit): ImagePickerLauncher
