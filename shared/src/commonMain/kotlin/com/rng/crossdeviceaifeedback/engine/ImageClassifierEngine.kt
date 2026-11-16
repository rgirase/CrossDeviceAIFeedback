package com.rng.crossdeviceaifeedback.engine

import com.rng.crossdeviceaifeedback.model.DeviceInfo
import com.rng.crossdeviceaifeedback.model.ExecutionBackend
import com.rng.crossdeviceaifeedback.model.InferenceRun
import com.rng.crossdeviceaifeedback.model.ModelArchitecture
import com.rng.crossdeviceaifeedback.model.PipelineMode

interface ImageClassifierEngine {
    fun getDeviceInfo(): DeviceInfo
    fun getAvailableBackends(): List<ExecutionBackend>

    suspend fun classify(
        sampleInputName: String,
        inputPixels: FloatArray,
        backend: ExecutionBackend,
        modelArchitecture: ModelArchitecture = ModelArchitecture.MOBILENET_V2_FP32,
        pipelineMode: PipelineMode = PipelineMode.PURE_MODEL_TENSOR
    ): InferenceRun
}

expect fun createPlatformClassifierEngine(): ImageClassifierEngine
