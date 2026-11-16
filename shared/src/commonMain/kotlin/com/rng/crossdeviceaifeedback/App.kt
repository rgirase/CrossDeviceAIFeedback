package com.rng.crossdeviceaifeedback

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.rng.crossdeviceaifeedback.engine.ModelDownloadManager
import com.rng.crossdeviceaifeedback.engine.createPlatformClassifierEngine
import com.rng.crossdeviceaifeedback.export.CloudSyncService
import com.rng.crossdeviceaifeedback.export.ResearchDataExporter
import com.rng.crossdeviceaifeedback.export.rememberFileSharer
import com.rng.crossdeviceaifeedback.metrics.ModelComparisonEngine
import com.rng.crossdeviceaifeedback.model.ComparisonResult
import com.rng.crossdeviceaifeedback.model.ExecutionBackend
import com.rng.crossdeviceaifeedback.model.InferenceRun
import com.rng.crossdeviceaifeedback.model.ModelArchitecture
import com.rng.crossdeviceaifeedback.model.PipelineMode
import com.rng.crossdeviceaifeedback.ui.decodeImageToNormalizedFloatArray
import com.rng.crossdeviceaifeedback.ui.rememberImagePickerLauncher
import kotlinx.coroutines.launch
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun App() {
    val engine = remember { createPlatformClassifierEngine() }
    val deviceInfo = remember { engine.getDeviceInfo() }
    val availableBackends = remember { engine.getAvailableBackends() }
    val fileSharer = rememberFileSharer()

    var selectedBackend by remember { mutableStateOf(availableBackends.firstOrNull() ?: ExecutionBackend.CPU_FP32) }
    var selectedPipelineMode by remember { mutableStateOf(PipelineMode.PURE_MODEL_TENSOR) }
    var selectedModelArchitecture by remember { mutableStateOf(ModelArchitecture.MOBILENET_V2_FP32) }

    var serverUrlInput by remember { mutableStateOf(CloudSyncService.targetServerUrl) }
    var autoSyncEnabled by remember { mutableStateOf(false) }

    // Dynamic Model Download States
    var modelUrlInput by remember { mutableStateOf("https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v2_1.0_224_quant.tflite") }
    var downloadProgress by remember { mutableStateOf(0.0f) }
    var isDownloadingModel by remember { mutableStateOf(false) }
    var downloadStatusMsg by remember { mutableStateOf<String?>(null) }
    var isModelReady by remember { mutableStateOf(ModelDownloadManager.isModelDownloaded()) }

    val sampleInputs = listOf(
        "Sample Image: Golden Retriever (224x224)",
        "Sample Image: Sports Car (224x224)",
        "Synthetic Standard Tensor #1"
    )
    var selectedInputName by remember { mutableStateOf(sampleInputs.first()) }
    var uploadedImageBytes by remember { mutableStateOf<ByteArray?>(null) }

    var runsHistory by remember { mutableStateOf(listOf<InferenceRun>()) }
    var comparisonResult by remember { mutableStateOf<ComparisonResult?>(null) }
    var isRunning by remember { mutableStateOf(false) }

    val coroutineScope = rememberCoroutineScope()
    val scrollState = rememberScrollState()

    val jsonEncoder = remember { Json { prettyPrint = true } }

    MaterialTheme {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp)
                    .verticalScroll(scrollState),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Header
                Text(
                    text = "AI Model Cross-Device Equivalence Benchmark",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )

                // Device Hardware Info Card
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text("Current Device Specs", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        Spacer(modifier = Modifier.height(4.dp))
                        Text("Platform: ${deviceInfo.platformName}", style = MaterialTheme.typography.bodyMedium)
                        Text("Model: ${deviceInfo.deviceModel}", style = MaterialTheme.typography.bodyMedium)
                        Text("OS: ${deviceInfo.osVersion}", style = MaterialTheme.typography.bodyMedium)
                        Text("Hardware/Chipset: ${deviceInfo.chipsetHardware}", style = MaterialTheme.typography.bodyMedium)
                    }
                }

                // Dynamic On-Device Model Download Card
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                ) {
                    Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text("⬇️ Dynamic Model Download Manager", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                        
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("Local Model File Status: ", style = MaterialTheme.typography.bodySmall)
                            Text(
                                text = if (isModelReady) "✅ Downloaded & Ready" else "⚠️ Model File Missing on Device",
                                style = MaterialTheme.typography.bodySmall,
                                fontWeight = FontWeight.Bold,
                                color = if (isModelReady) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error
                            )
                        }

                        OutlinedTextField(
                            value = modelUrlInput,
                            onValueChange = { modelUrlInput = it },
                            label = { Text("Model Download HTTPS URL (.tflite)") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth()
                        )

                        if (isDownloadingModel) {
                            Column(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                                Text("Downloading Model... ${(downloadProgress * 100).toInt()}%", style = MaterialTheme.typography.bodySmall)
                                LinearProgressIndicator(
                                    progress = { downloadProgress },
                                    modifier = Modifier.fillMaxWidth().height(6.dp)
                                )
                            }
                        }

                        Button(
                            onClick = {
                                coroutineScope.launch {
                                    isDownloadingModel = true
                                    downloadProgress = 0.0f
                                    val (success, msg) = ModelDownloadManager.downloadModel(modelUrlInput) { prog ->
                                        downloadProgress = prog
                                    }
                                    isDownloadingModel = false
                                    isModelReady = success
                                    downloadStatusMsg = msg
                                }
                            },
                            enabled = !isDownloadingModel,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("⬇️ Download Model File to Device Sandbox")
                        }

                        downloadStatusMsg?.let { msg ->
                            Text(msg, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
                        }
                    }
                }

                // Central Cloud Sync Setup Card
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
                ) {
                    Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text("☁️ Common Location Cloud Sync Setup", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                        Text("Set a shared Webhook / Google Sheets / Cloud REST endpoint URL for all devices:", style = MaterialTheme.typography.bodySmall)

                        OutlinedTextField(
                            value = serverUrlInput,
                            onValueChange = {
                                serverUrlInput = it
                                CloudSyncService.targetServerUrl = it
                            },
                            label = { Text("Central Server / Google Sheets Webhook URL") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth()
                        )

                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("⚡ Auto-push run to common location when completed", style = MaterialTheme.typography.bodySmall)
                            Switch(
                                checked = autoSyncEnabled,
                                onCheckedChange = { autoSyncEnabled = it }
                            )
                        }
                    }
                }

                // Benchmark Controls Card
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("Benchmark Setup", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)

                        // 1. Pipeline Mode Selector
                        Text("Pipeline Evaluation Mode:", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Bold)
                        PipelineMode.entries.forEach { mode ->
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                RadioButton(
                                    selected = (mode == selectedPipelineMode),
                                    onClick = { selectedPipelineMode = mode }
                                )
                                Column {
                                    Text(mode.displayName, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
                                    Text(mode.description, style = MaterialTheme.typography.bodySmall, fontSize = 11.sp, color = Color.Gray)
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(4.dp))
                        // 2. AI Model Selection
                        Text("Model Architecture:", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Bold)
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState())
                        ) {
                            ModelArchitecture.entries.forEach { model ->
                                FilterChip(
                                    selected = (model == selectedModelArchitecture),
                                    onClick = { selectedModelArchitecture = model },
                                    label = { Text(model.displayName, fontSize = 12.sp) }
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(4.dp))
                        // 3. Sample Input Picker
                        Text("Sample Input:", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Bold)
                        sampleInputs.forEach { inputName ->
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                RadioButton(
                                    selected = (inputName == selectedInputName),
                                    onClick = {
                                        selectedInputName = inputName
                                        uploadedImageBytes = null
                                    }
                                )
                                Text(inputName, style = MaterialTheme.typography.bodyMedium)
                            }
                        }

                        val imagePicker = rememberImagePickerLauncher { bytes, filename ->
                            if (bytes != null && filename != null) {
                                uploadedImageBytes = bytes
                                selectedInputName = "Custom Upload: $filename"
                            }
                        }

                        OutlinedButton(
                            onClick = { imagePicker.launch() },
                            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)
                        ) {
                            Text("📷 Upload Custom Image from Gallery")
                        }

                        Spacer(modifier = Modifier.height(4.dp))
                        // 4. Execution Backend Filter Chips
                        Text("Execution Backend:", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Bold)
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState())
                        ) {
                            availableBackends.forEach { backend ->
                                FilterChip(
                                    selected = (backend == selectedBackend),
                                    onClick = { selectedBackend = backend },
                                    label = { Text(backend.displayName, fontSize = 12.sp) }
                                )
                            }
                        }

                        Spacer(modifier = Modifier.height(8.dp))
                        Button(
                            onClick = {
                                coroutineScope.launch {
                                    isRunning = true

                                    val inputPixels = if (uploadedImageBytes != null) {
                                        decodeImageToNormalizedFloatArray(uploadedImageBytes!!, 224, 224)
                                    } else if (selectedInputName.contains("Golden Retriever", ignoreCase = true)) {
                                        FloatArray(3 * 224 * 224) { i -> ((i * 17) % 255 - 127.5f) / 127.5f }
                                    } else if (selectedInputName.contains("Sports Car", ignoreCase = true)) {
                                        FloatArray(3 * 224 * 224) { i -> ((i * 31) % 255 - 127.5f) / 127.5f }
                                    } else {
                                        FloatArray(3 * 224 * 224) { 0.5f }
                                    }

                                    val newRun = engine.classify(
                                        sampleInputName = selectedInputName,
                                        inputPixels = inputPixels,
                                        backend = selectedBackend,
                                        modelArchitecture = selectedModelArchitecture,
                                        pipelineMode = selectedPipelineMode
                                    )

                                    val updatedHistory = listOf(newRun) + runsHistory
                                    runsHistory = updatedHistory

                                    if (updatedHistory.size >= 2) {
                                        comparisonResult = ModelComparisonEngine.compareRuns(
                                            runA = updatedHistory[0],
                                            runB = updatedHistory[1]
                                        )
                                    }

                                    if (autoSyncEnabled) {
                                        CloudSyncService.syncRunToCentralServer(newRun)
                                    }

                                    isRunning = false
                                }
                            },
                            enabled = !isRunning,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            if (isRunning) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(18.dp),
                                    color = MaterialTheme.colorScheme.onPrimary,
                                    strokeWidth = 2.dp
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Running Single Inference...")
                            } else {
                                Text("Run Single Model Inference")
                            }
                        }

                        OutlinedButton(
                            enabled = !isRunning,
                            onClick = {
                                coroutineScope.launch {
                                    isRunning = true

                                    val inputPixels = if (uploadedImageBytes != null) {
                                        decodeImageToNormalizedFloatArray(uploadedImageBytes!!, 224, 224)
                                    } else if (selectedInputName.contains("Golden Retriever", ignoreCase = true)) {
                                        FloatArray(3 * 224 * 224) { i -> ((i * 17) % 255 - 127.5f) / 127.5f }
                                    } else if (selectedInputName.contains("Sports Car", ignoreCase = true)) {
                                        FloatArray(3 * 224 * 224) { i -> ((i * 31) % 255 - 127.5f) / 127.5f }
                                    } else {
                                        FloatArray(3 * 224 * 224) { 0.5f }
                                    }

                                    val newMatrixRuns = mutableListOf<InferenceRun>()
                                    for (mode in listOf(PipelineMode.PURE_MODEL_TENSOR, PipelineMode.END_TO_END_DECODED)) {
                                        for (b in availableBackends) {
                                            val run = engine.classify(
                                                sampleInputName = selectedInputName,
                                                inputPixels = inputPixels,
                                                backend = b,
                                                modelArchitecture = selectedModelArchitecture,
                                                pipelineMode = mode
                                            )
                                            newMatrixRuns.add(run)
                                            if (autoSyncEnabled) {
                                                CloudSyncService.syncRunToCentralServer(run)
                                            }
                                        }
                                    }

                                    val updatedHistory = newMatrixRuns + runsHistory
                                    runsHistory = updatedHistory

                                    if (updatedHistory.size >= 2) {
                                        comparisonResult = ModelComparisonEngine.compareRuns(
                                            runA = updatedHistory[0],
                                            runB = updatedHistory[1]
                                        )
                                    }

                                    isRunning = false
                                }
                            },
                            modifier = Modifier.fillMaxWidth().padding(top = 4.dp)
                        ) {
                            Text("⚡ Run Full IEEE Experiment Matrix (All Modes & Backends)")
                        }
                    }
                }

                // Equivalence Metrics Card (if at least 2 runs exist)
                comparisonResult?.let { result ->
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.tertiaryContainer)
                    ) {
                        Column(
                            modifier = Modifier.padding(12.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Text(
                                text = "Equivalence Metrics (Latest Run vs Previous)",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onTertiaryContainer
                            )
                            HorizontalDivider()

                            val runA = runsHistory.getOrNull(0)
                            val runB = runsHistory.getOrNull(1)

                            if (runA != null && runB != null) {
                                val modeText = if (runA.pipelineMode == runB.pipelineMode) {
                                    "Mode: ${runA.pipelineMode.displayName}"
                                } else {
                                    "Comparing Mode A (${runA.pipelineMode.name}) vs Mode B (${runB.pipelineMode.name})"
                                }
                                Text(modeText, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold)
                            }

                            MetricRow("Cosine Similarity", "${(result.cosineSimilarity * 100).format(4)}%")
                            MetricRow("Mean Absolute Error (MAE)", result.meanAbsoluteError.format(6))
                            MetricRow("Max Absolute Delta (Δ max)", result.maxAbsoluteDelta.format(6))
                            MetricRow("KL Divergence", result.klDivergence.format(6))
                            MetricRow("Top-1 Output Class Match", if (result.top1Matches) "EXACT MATCH (100%)" else "MISMATCH")
                            MetricRow("Top-5 Class Overlap", "${(result.top5OverlapRatio * 100).toInt()}%")
                        }
                    }
                }

                // Latest Inference Run Card
                runsHistory.firstOrNull()?.let { latestRun ->
                    Card(modifier = Modifier.fillMaxWidth()) {
                        Column(
                            modifier = Modifier.padding(12.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Text("Latest Run Result", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                            Text("Framework: ${latestRun.engineFramework}", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.SemiBold)
                            Text("Model: ${latestRun.modelName}", style = MaterialTheme.typography.bodyMedium)
                            Text("Backend: ${latestRun.backend.displayName}", style = MaterialTheme.typography.bodyMedium)
                            Text("Pipeline Mode: ${latestRun.pipelineMode.displayName}", style = MaterialTheme.typography.bodyMedium)
                            Text("Inference Latency: ${latestRun.inferenceTimeMs.format(2)} ms", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)

                            Spacer(modifier = Modifier.height(4.dp))
                            Text("Top Predictions:", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold)

                            latestRun.topKPredictions.forEach { pred ->
                                Column(modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween
                                    ) {
                                        Text("${pred.classIndex}: ${pred.label}", style = MaterialTheme.typography.bodySmall)
                                        Text("${(pred.probability * 100).format(2)}%", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Bold)
                                    }
                                    LinearProgressIndicator(
                                        progress = { pred.probability },
                                        modifier = Modifier.fillMaxWidth().height(6.dp),
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(4.dp))
                            var syncStatusMessage by remember { mutableStateOf<String?>(null) }
                            Button(
                                onClick = {
                                    val (_, msg) = CloudSyncService.syncRunToCentralServer(latestRun)
                                    syncStatusMessage = msg
                                },
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("☁️ Push to Common Cloud Location")
                            }

                            syncStatusMessage?.let { msg ->
                                Text(
                                    text = msg,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.primary,
                                    fontWeight = FontWeight.SemiBold
                                )
                            }

                            OutlinedButton(
                                onClick = {
                                    val jsonPayload = jsonEncoder.encodeToString(latestRun)
                                    fileSharer.shareFile("${latestRun.id}.json", jsonPayload)
                                },
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("📤 Share JSON Run Log (Google Drive / AirDrop / Email)")
                            }

                            Spacer(modifier = Modifier.height(8.dp))
                            Text("Export Log (JSON):", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold)
                            val jsonText = jsonEncoder.encodeToString(latestRun)
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(180.dp)
                                    .background(Color.Black.copy(alpha = 0.05f), shape = RoundedCornerShape(4.dp))
                                    .border(1.dp, Color.Gray.copy(alpha = 0.3f), shape = RoundedCornerShape(4.dp))
                                    .padding(8.dp)
                                    .verticalScroll(rememberScrollState())
                            ) {
                                Text(
                                    text = jsonText,
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 10.sp,
                                    modifier = Modifier.fillMaxWidth()
                                )
                            }
                        }
                    }
                }

                // Aggregated CSV Dataset Export Card
                if (runsHistory.isNotEmpty()) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)
                    ) {
                        Column(
                            modifier = Modifier.padding(12.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Text("Aggregated CSV Research Log (${runsHistory.size} Runs)", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                            Text("Copy or push CSV dataset for Python/Jupyter graph generation:", style = MaterialTheme.typography.bodySmall)

                            val csvText = remember(runsHistory) { ResearchDataExporter.exportToCsv(runsHistory) }

                            Button(
                                onClick = {
                                    fileSharer.shareFile("cross_device_benchmark_dataset.csv", csvText)
                                },
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("📤 Share CSV Dataset File (Share Sheet)")
                            }

                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(180.dp)
                                    .background(Color.Black.copy(alpha = 0.05f), shape = RoundedCornerShape(4.dp))
                                    .border(1.dp, Color.Gray.copy(alpha = 0.3f), shape = RoundedCornerShape(4.dp))
                                    .padding(8.dp)
                                    .verticalScroll(rememberScrollState())
                            ) {
                                Text(
                                    text = csvText,
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 10.sp,
                                    modifier = Modifier.fillMaxWidth()
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun MetricRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Text(value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Bold)
    }
}

private fun Float.format(digits: Int): String {
    val factor = when(digits) {
        2 -> 100.0
        4 -> 10000.0
        else -> 1000000.0
    }
    val rounded = (this * factor).toInt() / factor
    return rounded.toString()
}

private fun Double.format(digits: Int): String {
    val factor = when(digits) {
        2 -> 100.0
        4 -> 10000.0
        else -> 1000000.0
    }
    val rounded = (this * factor).toInt() / factor
    return rounded.toString()
}
