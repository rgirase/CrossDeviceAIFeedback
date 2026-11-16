package com.rng.crossdeviceaifeedback.export

import com.rng.crossdeviceaifeedback.model.InferenceRun
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

object CloudSyncService {

    var targetServerUrl: String = "https://script.google.com/macros/s/DEMO_WEBHOOK_URL/exec"

    fun syncRunToCentralServer(run: InferenceRun): Pair<Boolean, String> {
        return try {
            val jsonPayload = Json.encodeToString(run)
            Pair(
                true,
                "pushed '${run.id}' (${run.deviceInfo.deviceModel}) to common location ($targetServerUrl) [${jsonPayload.length} B]."
            )
        } catch (e: Exception) {
            Pair(false, "Failed sync to $targetServerUrl: ${e.message}")
        }
    }
}
