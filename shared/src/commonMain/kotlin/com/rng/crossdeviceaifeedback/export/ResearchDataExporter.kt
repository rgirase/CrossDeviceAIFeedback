package com.rng.crossdeviceaifeedback.export

import com.rng.crossdeviceaifeedback.model.InferenceRun

object ResearchDataExporter {

    fun exportToCsv(runs: List<InferenceRun>): String {
        val sb = StringBuilder()
        // CSV Header
        sb.append("RunID,Timestamp,Platform,DeviceModel,OSVersion,Chipset,ModelName,SampleInput,Backend,LatencyMs,Top1ClassIndex,Top1Label,Top1Probability\n")

        for (run in runs) {
            val top1 = run.topKPredictions.firstOrNull()
            val top1Class = top1?.classIndex ?: -1
            val top1Label = top1?.label?.replace(",", " ") ?: "None"
            val top1Prob = top1?.probability ?: 0.0f

            sb.append("${run.id},")
              .append("${run.timestampMs},")
              .append("${run.deviceInfo.platformName},")
              .append("\"${run.deviceInfo.deviceModel}\",")
              .append("\"${run.deviceInfo.osVersion}\",")
              .append("\"${run.deviceInfo.chipsetHardware}\",")
              .append("\"${run.modelName}\",")
              .append("\"${run.sampleInputName}\",")
              .append("${run.backend.name},")
              .append("${run.inferenceTimeMs},")
              .append("$top1Class,")
              .append("\"$top1Label\",")
              .append("$top1Prob\n")
        }
        return sb.toString()
    }
}
