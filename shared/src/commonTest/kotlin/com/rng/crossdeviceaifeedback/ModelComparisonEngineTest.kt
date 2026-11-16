package com.rng.crossdeviceaifeedback

import com.rng.crossdeviceaifeedback.metrics.ModelComparisonEngine
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ModelComparisonEngineTest {

    @Test
    fun testCosineSimilarityIdenticalVectors() {
        val vec1 = floatArrayOf(0.1f, 0.4f, 0.3f, 0.2f)
        val vec2 = floatArrayOf(0.1f, 0.4f, 0.3f, 0.2f)

        val sim = ModelComparisonEngine.cosineSimilarity(vec1, vec2)
        assertTrue(abs(sim - 1.0f) < 1e-5f, "Cosine similarity of identical vectors must be 1.0")
    }

    @Test
    fun testMeanAbsoluteErrorAndMaxDelta() {
        val vecA = floatArrayOf(0.5f, 0.2f, 0.8f)
        val vecB = floatArrayOf(0.4f, 0.2f, 0.6f)

        val mae = ModelComparisonEngine.meanAbsoluteError(vecA, vecB)
        val maxDelta = ModelComparisonEngine.maxAbsoluteDelta(vecA, vecB)

        // Differences: |0.5-0.4|=0.1, |0.2-0.2|=0.0, |0.8-0.6|=0.2. MAE = (0.1+0.0+0.2)/3 = 0.1
        assertTrue(abs(mae - 0.1f) < 1e-5f, "MAE should be 0.1")
        assertTrue(abs(maxDelta - 0.2f) < 1e-5f, "Max Delta should be 0.2")
    }

    @Test
    fun testKLDivergenceIdenticalDistributions() {
        val p = floatArrayOf(0.2f, 0.5f, 0.3f)
        val q = floatArrayOf(0.2f, 0.5f, 0.3f)

        val kl = ModelComparisonEngine.klDivergence(p, q)
        assertTrue(abs(kl) < 1e-4f, "KL divergence of identical distributions must be ~0.0")
    }

    @Test
    fun testTopKIndicesSorting() {
        val probs = floatArrayOf(0.05f, 0.70f, 0.10f, 0.15f)
        val top2 = ModelComparisonEngine.topKIndices(probs, 2)

        assertEquals(listOf(1, 3), top2, "Top 2 indices should be [1, 3]")
    }
}
