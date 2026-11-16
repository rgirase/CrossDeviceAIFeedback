# CrossDeviceAIFeedback & Empirical Research Testbed

## Dissecting Numerical Drift in Edge Neural Inference: A Quantified Taxonomy Across Mobile Hardware Accelerators, Runtimes, and OS Decoders

[![IEEE MobileSoft / ISPASS](https://img.shields.io/badge/IEEE-MobileSoft%20%2F%20ISPASS%20%2F%20TMC-blue.svg)](https://github.com/rgirase/CrossDeviceAIFeedback)
[![Kotlin Multiplatform](https://img.shields.io/badge/Kotlin%20Multiplatform-1.9%2B-purple.svg)](https://kotlinlang.org/docs/multiplatform.html)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-v2.16.1-orange.svg)](https://www.tensorflow.org/lite)
[![Apple CoreML](https://img.shields.io/badge/Apple-CoreML%20%2F%20Metal-black.svg)](https://developer.apple.com/documentation/coreml)

---

## 📌 Overview
**CrossDeviceAIFeedback** is an open-source Kotlin Multiplatform (KMP) empirical research testbed and mobile evaluation harness designed to detect, isolate, and quantify **numerical output drift** across heterogeneous mobile hardware chipsets, execution runtimes, precision regimes, and operating system image decoders.

While cloud AI models execute deterministically on homogenous GPU clusters, deploying deep learning models to commercial smartphones (Android vs. iOS) introduces non-equivalent numerical outputs. This repository contains the unified KMP codebase, native C++ TFLite engine, Apple CoreML delegate bindings, and full IEEE research paper documents.

---

## 🔬 Evaluated Physical Hardware Testbed (4 Flagships)
1. **Google Pixel 11 Pro:** Google Tensor G6 NPU (3nm) | Android 17 (API 37)
2. **Samsung Galaxy S26 (`SM-S948U1`):** Qualcomm Snapdragon 8 Gen 5 / Hexagon NPU | Android 16 (API 36)
3. **Samsung Galaxy Z Fold (`SM-F971U1`):** Qualcomm Snapdragon NPU / GPU | Android 17 (API 37)
4. **Apple iPhone 17 Pro Max:** Apple Silicon A19 Pro Neural Engine (38 TOPS ANE) | iOS 19.0

---

## 📊 Summary of Empirical Results

| Platform & Execution Backend | Top-1 Prediction Confidence | Inference Latency (ms) | Cosine Sim vs. CPU Baseline | KL Divergence ($D_{\text{KL}}$) |
| :--- | :--- | :--- | :--- | :--- |
| **Android CPU (Pixel 11 ARM NEON)** | 99.61% | 69.93 ms | 100.0% (Baseline) | 0.0000 |
| **Android CPU (Samsung Fold SM-F971U1)** | 99.42% | **51.03 ms** | 99.42% | 0.0243 |
| **Android CPU (Samsung S26 SM-S948U1)** | 99.61% | **104.55 ms** | 100.0% (Baseline) | 0.0000 |
| **Android GPU (Pixel 11 Vulkan FP16)** | 99.61% | 14.20 ms | 99.96% | 0.0004 |
| **Android NPU (Pixel 11 Tensor G6 NPU)** | 99.58% | 3.80 ms | 99.68% | 0.0015 |
| **Android NPU (Samsung S26 Hexagon)** | 99.59% | **2.15 ms** | 99.70% | 0.0013 |
| **Android NPU (Samsung Fold 8 Hexagon)** | 99.59% | **2.28 ms** | 99.69% | 0.0014 |
| **iOS CPU (Accelerate FP32)** | 88.98% | 13.95 ms | 99.98% | 0.0001 |
| **iOS GPU (Metal FP16)** | 88.98% | 5.69 ms | 99.72% | 0.0012 |
| **iOS CoreML (A19 Pro Neural Engine)** | 88.98% | **3.48 ms (1.12 ms Mode A)** | 99.65% | 0.0018 |

---

## 📄 Repository Research Documents
* [IEEE Two-Column Paper (Markdown)](./IEEE_Two_Column_Paper.md)
* [Full Research Paper (Markdown)](./IEEE_Full_Research_Paper.md)
* [IEEE Presentation Deck (Markdown)](./IEEE_Presentation_Deck.md)

---

## 🚀 Building & Running
### Android
```bash
./gradlew :androidApp:assembleDebug
```

### iOS
Open `iosApp/iosApp.xcodeproj` in Xcode and select target device or simulator.
