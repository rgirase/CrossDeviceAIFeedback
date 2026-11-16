package com.rng.crossdeviceaifeedback

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform