package com.rng.crossdeviceaifeedback.export

import androidx.compose.runtime.Composable

interface FileSharer {
    fun shareFile(filename: String, content: String)
}

@Composable
expect fun rememberFileSharer(): FileSharer
