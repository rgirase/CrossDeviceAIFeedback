package com.rng.crossdeviceaifeedback.model

import kotlinx.serialization.Serializable

@Serializable
enum class PipelineMode(val displayName: String, val description: String) {
    PURE_MODEL_TENSOR(
        displayName = "Pure Float32 Tensor (Mode A)",
        description = "Bypasses OS image decoders. Feeds deterministic Float32 tensor matrix directly into model to isolate floating-point inference drift."
    ),
    END_TO_END_DECODED(
        displayName = "End-to-End Image Decoding (Mode B)",
        description = "Decodes JPEG/PNG image natively (Android Skia vs. iOS CoreGraphics), resizes to 224x224, and measures combined decoder drift + model drift."
    )
}

@Serializable
enum class ModelArchitecture(
    val displayName: String,
    val layerDepthL: Int,
    val quantizationBitsQ: Int,
    val operatorSensitivitySO: Float
) {
    MOBILENET_V2_FP32("MobileNetV2 FP32", layerDepthL = 16, quantizationBitsQ = 32, operatorSensitivitySO = 1.24f),
    MOBILENET_V2_INT8("MobileNetV2 INT8 Quantized", layerDepthL = 16, quantizationBitsQ = 8, operatorSensitivitySO = 1.24f),
    RESNET_50_FP32("ResNet-50 FP32", layerDepthL = 50, quantizationBitsQ = 32, operatorSensitivitySO = 2.45f),
    SQUEEZENET_FP32("SqueezeNet FP32", layerDepthL = 18, quantizationBitsQ = 32, operatorSensitivitySO = 0.85f)
}

@Serializable
enum class ExecutionBackend(val displayName: String) {
    CPU_FP32("CPU (Float32)"),
    GPU_FP16("GPU Delegate (FP16/FP32)"),
    NNAPI_NPU("Android NNAPI / NPU"),
    CORE_ML("Apple CoreML / Neural Engine")
}

@Serializable
data class DeviceInfo(
    val platformName: String,
    val deviceModel: String,
    val osVersion: String,
    val chipsetHardware: String = "Unknown"
)

@Serializable
data class ClassificationPrediction(
    val classIndex: Int,
    val label: String,
    val probability: Float
)

@Serializable
data class InferenceRun(
    val id: String,
    val modelName: String,
    val sampleInputName: String,
    val deviceInfo: DeviceInfo,
    val backend: ExecutionBackend,
    val pipelineMode: PipelineMode = PipelineMode.PURE_MODEL_TENSOR,
    val modelArchitecture: ModelArchitecture = ModelArchitecture.MOBILENET_V2_FP32,
    val layerDepthL: Int = 16,
    val quantizationBitsQ: Int = 32,
    val operatorSensitivitySO: Float = 1.24f,
    val engineFramework: String = "TensorFlow Lite Native Runtime (v2.16.1)",
    val rawProbabilities: List<Float>,
    val topKPredictions: List<ClassificationPrediction>,
    val inferenceTimeMs: Double,
    val timestampMs: Long
)

@Serializable
data class ComparisonResult(
    val runAId: String,
    val runBId: String,
    val deviceA: DeviceInfo,
    val deviceB: DeviceInfo,
    val backendA: ExecutionBackend,
    val backendB: ExecutionBackend,
    val cosineSimilarity: Float,
    val meanAbsoluteError: Float,
    val maxAbsoluteDelta: Float,
    val klDivergence: Float,
    val top1Matches: Boolean,
    val top5OverlapRatio: Float
)
