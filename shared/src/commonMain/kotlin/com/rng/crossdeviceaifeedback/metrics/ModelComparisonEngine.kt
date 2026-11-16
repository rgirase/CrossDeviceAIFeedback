package com.rng.crossdeviceaifeedback.metrics

import com.rng.crossdeviceaifeedback.model.ComparisonResult
import com.rng.crossdeviceaifeedback.model.InferenceRun
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.sqrt

object ModelComparisonEngine {

    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size && a.isNotEmpty()) { "Vectors must be non-empty and equal in length." }
        var dotProduct = 0.0
        var normA = 0.0
        var normB = 0.0
        for (i in a.indices) {
            val valA = a[i].toDouble()
            val valB = b[i].toDouble()
            dotProduct += valA * valB
            normA += valA * valA
            normB += valB * valB
        }
        val denominator = sqrt(normA) * sqrt(normB)
        return if (denominator == 0.0) 1.0f else (dotProduct / denominator).toFloat()
    }

    fun meanAbsoluteError(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size && a.isNotEmpty()) { "Vectors must be non-empty and equal in length." }
        var sumDiff = 0.0
        for (i in a.indices) {
            sumDiff += abs(a[i].toDouble() - b[i].toDouble())
        }
        return (sumDiff / a.size).toFloat()
    }

    fun maxAbsoluteDelta(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size && a.isNotEmpty()) { "Vectors must be non-empty and equal in length." }
        var maxDelta = 0.0f
        for (i in a.indices) {
            val delta = abs(a[i] - b[i])
            if (delta > maxDelta) {
                maxDelta = delta
            }
        }
        return maxDelta
    }

    fun klDivergence(p: FloatArray, q: FloatArray, epsilon: Float = 1e-7f): Float {
        require(p.size == q.size && p.isNotEmpty()) { "Vectors must be non-empty and equal in length." }
        var kl = 0.0
        for (i in p.indices) {
            val pClamped = p[i].coerceAtLeast(epsilon).toDouble()
            val qClamped = q[i].coerceAtLeast(epsilon).toDouble()
            kl += pClamped * ln(pClamped / qClamped)
        }
        return kl.toFloat()
    }

    fun topKIndices(a: FloatArray, k: Int = 5): List<Int> {
        val effectiveK = k.coerceAtMost(a.size)
        return a.indices.sortedByDescending { a[it] }.take(effectiveK)
    }

    fun compareRuns(runA: InferenceRun, runB: InferenceRun): ComparisonResult {
        val vectorA = runA.rawProbabilities.toFloatArray()
        val vectorB = runB.rawProbabilities.toFloatArray()

        val cosSim = cosineSimilarity(vectorA, vectorB)
        val mae = meanAbsoluteError(vectorA, vectorB)
        val maxDelta = maxAbsoluteDelta(vectorA, vectorB)
        val klDiv = klDivergence(vectorA, vectorB)

        val top5A = topKIndices(vectorA, 5)
        val top5B = topKIndices(vectorB, 5)

        val top1Match = top5A.firstOrNull() == top5B.firstOrNull()
        val overlapCount = top5A.intersect(top5B.toSet()).size
        val overlapRatio = if (top5A.isEmpty()) 1.0f else overlapCount.toFloat() / top5A.size

        return ComparisonResult(
            runAId = runA.id,
            runBId = runB.id,
            deviceA = runA.deviceInfo,
            deviceB = runB.deviceInfo,
            backendA = runA.backend,
            backendB = runB.backend,
            cosineSimilarity = cosSim,
            meanAbsoluteError = mae,
            maxAbsoluteDelta = maxDelta,
            klDivergence = klDiv,
            top1Matches = top1Match,
            top5OverlapRatio = overlapRatio
        )
    }
}
