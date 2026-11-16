package com.rng.crossdeviceaifeedback.engine

import com.rng.crossdeviceaifeedback.model.ClassificationPrediction
import com.rng.crossdeviceaifeedback.model.DeviceInfo
import com.rng.crossdeviceaifeedback.model.ExecutionBackend
import com.rng.crossdeviceaifeedback.model.InferenceRun
import com.rng.crossdeviceaifeedback.model.ModelArchitecture
import com.rng.crossdeviceaifeedback.model.PipelineMode
import com.rng.crossdeviceaifeedback.metrics.ModelComparisonEngine
import kotlin.math.abs
import kotlin.math.exp
import kotlin.random.Random

class SyntheticClassifierEngine(
    private val simulatedDeviceInfo: DeviceInfo
) : ImageClassifierEngine {

    private val sampleLabels = listOf(
        "Golden Retriever", "Tabby Cat", "Sports Car", "Acoustic Guitar", "Coffee Mug",
        "Laptop", "Smartphone", "Mountain Bike", "Running Shoe", "Airplane",
        "Panda", "Penguin", "Sunflower", "Violin", "Pizza"
    )

    override fun getDeviceInfo(): DeviceInfo = simulatedDeviceInfo

    override fun getAvailableBackends(): List<ExecutionBackend> {
        return listOf(
            ExecutionBackend.CPU_FP32,
            ExecutionBackend.GPU_FP16,
            if (simulatedDeviceInfo.platformName == "Android") ExecutionBackend.NNAPI_NPU else ExecutionBackend.CORE_ML
        )
    }

    override suspend fun classify(
        sampleInputName: String,
        inputPixels: FloatArray,
        backend: ExecutionBackend,
        modelArchitecture: ModelArchitecture,
        pipelineMode: PipelineMode
    ): InferenceRun {
        val numClasses = 1000
        val rawLogits = FloatArray(numClasses)

        // Seed STRICTLY derived from the mathematical sum and variance of actual raw RGB image pixel float values
        var pixelSum = 0.0f
        var pixelAbsSum = 0.0f
        for (f in inputPixels) {
            pixelSum += f
            pixelAbsSum += abs(f)
        }

        // Handle Int.MIN_VALUE overflow safely by converting to positive Long
        val rawHash = (pixelSum * 1000.0f + pixelAbsSum * 50000.0f).toLong()
        val positiveSeed = abs(rawHash) + 101L
        val random = Random(positiveSeed.toInt())

        for (i in 0 until numClasses) {
            rawLogits[i] = random.nextFloat() * 2.0f - 1.0f
        }

        // Ensure category index is strictly positive and within valid bounds [0, sampleLabels.size - 1]
        val targetCategoryIdx = (positiveSeed % sampleLabels.size).toInt()
        val idx0 = (targetCategoryIdx % numClasses + numClasses) % numClasses
        val idx1 = ((targetCategoryIdx + 1) % numClasses + numClasses) % numClasses
        val idx2 = ((targetCategoryIdx + 2) % numClasses + numClasses) % numClasses

        rawLogits[idx0] += 6.5f
        rawLogits[idx1] += 2.5f
        rawLogits[idx2] += 1.8f

        // Apply micro-jitter based on backend AND model architecture
        var jitterMagnitude = when (backend) {
            ExecutionBackend.CPU_FP32 -> 0.00001f
            ExecutionBackend.GPU_FP16 -> 0.0025f
            ExecutionBackend.NNAPI_NPU, ExecutionBackend.CORE_ML -> 0.0050f
        }

        if (modelArchitecture == ModelArchitecture.MOBILENET_V2_INT8) {
            jitterMagnitude += 0.0080f // INT8 quantization rounding noise
        }

        // Apply OS-specific JPEG/PNG Image Decoder Downsampling Drift when Mode B is selected
        if (pipelineMode == PipelineMode.END_TO_END_DECODED) {
            val decoderDrift = if (simulatedDeviceInfo.platformName == "Android") 0.0040f else 0.0065f
            val decoderRandom = Random((positiveSeed + 8888L).toInt())
            for (i in 0 until numClasses) {
                rawLogits[i] += (decoderRandom.nextFloat() - 0.5f) * 2.0f * decoderDrift
            }
        }

        val backendRandom = Random((positiveSeed + backend.ordinal * 997L).toInt())
        for (i in 0 until numClasses) {
            val jitter = (backendRandom.nextFloat() - 0.5f) * 2.0f * jitterMagnitude
            rawLogits[i] += jitter
        }

        // Compute Softmax probabilities
        val maxLogit = rawLogits.maxOrNull() ?: 0.0f
        var expSum = 0.0
        val expLogits = FloatArray(numClasses)
        for (i in 0 until numClasses) {
            val expVal = exp((rawLogits[i] - maxLogit).toDouble())
            expLogits[i] = expVal.toFloat()
            expSum += expVal
        }

        val probabilities = FloatArray(numClasses) { i -> (expLogits[i] / expSum).toFloat() }

        // Extract top-K predictions
        val topKIndices = ModelComparisonEngine.topKIndices(probabilities, 5)
        val topKPredictions = topKIndices.map { classIdx ->
            val labelName = if (classIdx < sampleLabels.size) sampleLabels[classIdx] else "ImageNet Category #$classIdx"
            ClassificationPrediction(
                classIndex = classIdx,
                label = labelName,
                probability = probabilities[classIdx]
            )
        }

        val modelMultiplier = when (modelArchitecture) {
            ModelArchitecture.MOBILENET_V2_FP32 -> 1.0
            ModelArchitecture.MOBILENET_V2_INT8 -> 0.65
            ModelArchitecture.RESNET_50_FP32 -> 2.4
            ModelArchitecture.SQUEEZENET_FP32 -> 0.45
        }

        val baseLatency = when (backend) {
            ExecutionBackend.CPU_FP32 -> 18.5
            ExecutionBackend.GPU_FP16 -> 6.2
            ExecutionBackend.NNAPI_NPU, ExecutionBackend.CORE_ML -> 3.8
        } * modelMultiplier

        return InferenceRun(
            id = "run_${simulatedDeviceInfo.platformName}_${backend.name}_${Random.nextInt(1000, 9999)}",
            modelName = "${modelArchitecture.displayName} (${pipelineMode.displayName})",
            sampleInputName = sampleInputName,
            deviceInfo = simulatedDeviceInfo,
            backend = backend,
            pipelineMode = pipelineMode,
            modelArchitecture = modelArchitecture,
            rawProbabilities = probabilities.toList(),
            topKPredictions = topKPredictions,
            inferenceTimeMs = baseLatency + (Random.nextDouble() * 1.5),
            timestampMs = 1710000000000L
        )
    }
}
