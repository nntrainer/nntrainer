<p align="center">
  <h1 align="center">NNTrainer</h1>
  <p align="center">
    <strong>On-Device AI Training & LLM Inference Framework</strong><br/>
    Run 30B+ LLMs on your phone. Train models without the cloud.
  </p>
</p>

<p align="center">
  <a href="https://github.com/nntrainer/nntrainer/actions/workflows/daily_build.yml"><img src="https://github.com/nntrainer/nntrainer/actions/workflows/daily_build.yml/badge.svg" alt="DailyBuild"/></a>
  <a href="https://nntrainer.github.io/coverage_result/"><img src="https://img.shields.io/endpoint?url=https://nntrainer.github.io/coverage_result/coverage.json" alt="Code Coverage"/></a>
  <a href="https://scan.coverity.com/projects/nnstreamer-nntrainer"><img src="https://scan.coverity.com/projects/22512/badge.svg" alt="Coverity Scan Build Status"/></a>
  <a href="https://www.bestpractices.dev/projects/9179"><img src="https://www.bestpractices.dev/projects/9179/badge" alt="OpenSSF Best Practices"/></a>
  <img src="https://img.shields.io/github/repo-size/nnstreamer/nntrainer" alt="GitHub repo size"/>
  <img src="https://img.shields.io/github/issues/nnstreamer/nntrainer" alt="GitHub issues"/>
  <img src="https://img.shields.io/github/issues-pr/nnstreamer/nntrainer" alt="GitHub pull requests"/>
</p>

---

## Run 30B MoE LLMs on a Mobile Phone

NNTrainer makes it possible to run large-scale Mixture-of-Experts LLMs directly on mobile devices using **Flash Storage Utilization (FSU)** — loading experts on-the-fly from flash storage instead of keeping the entire model in memory.

| GPT-OSS 20B on Mobile | Qwen3 MoE 30B-A3B on Mobile |
|:----------------------:|:----------------------------:|
| <img src="docs/videos/GPT_OSS_20B_Demo.gif" width="360"> | <img src="docs/videos/Qwen_30B_Demo.gif" width="360"> |

### Memory? Not a problem.

With FSU, NNTrainer loads only the active experts during inference — reducing memory from **16.5 GB down to 1.3 GB** for a 30B-parameter model.

| Load Whole Model (Qwen3-30B-A3B) | Load Experts On-The-Fly (Qwen3-30B-A3B-Slim) |
|:---------------------------------:|:---------------------------------------------:|
| ![](./docs/videos/moe-full.gif) | ![](./docs/videos/moe-on-the-fly.gif) |
| Memory: **16.5 GB** | Memory: **1.3 GB** |

> Try it yourself with `Applications/CausalLM/models/*-slim` models.

---

## Applications/CausalLM — LLM Inference Engine

[`Applications/CausalLM`](https://github.com/nntrainer/nntrainer/tree/main/Applications/CausalLM) is NNTrainer's production-ready LLM inference engine optimized for resource-constrained environments.

### Supported Models

| Model | Parameters | Variants |
|-------|-----------|----------|
| **Qwen3** | 0.6B, 1.7B, 4B, 8B, 14B, 32B | Standard |
| **Qwen3-MoE** | 30B-A3B | Full / Slim (FSU) / Cached-Slim |
| **GPT-OSS** | 20B-A3.6B, 120B-5.1B | Full / Cached-Slim |
| **Gemma3** | - | Standard |
| **Qwen2** | - | Standard |

### Core Optimizations

- **FSU (Flash Storage Utilization)** — Dynamically loads weights from flash storage during inference, dramatically reducing peak memory usage
- **MoE Cache** — Intelligent caching keeps frequently used experts in memory, swapping others to storage
- **Proactive Loading** — Predicts and pre-loads required weights before they are needed, minimizing latency
- **Decoupled KV Cache** — Separates query and KV cache computation for efficient attention in long-context scenarios

### Build & Deploy

CausalLM supports multiple deployment targets with ready-to-use build scripts:

```bash
cd Applications/CausalLM

# Android
./build_android.sh
# Add --app for the QuickDotAI AAR/APK and --install for device deployment.

# Linux / PC
meson build && ninja -C build
```

Includes benchmark tools with thermal monitoring, performance metrics (prefill/generation speed, peak memory), and Android JNI bindings.

---

## On-Device Training — Learn Directly on the Edge

NNTrainer was built from the ground up for **training neural networks on device** — no cloud, no data upload, no privacy risk. User data never leaves the device.

### Training Scenarios

#### Transfer Learning
Freeze a pre-trained backbone (e.g., MobileNetV2) and fine-tune only the final layers with your own data. Train a custom image classifier with just **15 images** in under a minute on a smartphone.

```
Pre-trained MobileNetV2 (frozen) → FC(128) → FC(20) → Softmax(3)
                                   ↑ Only these layers are trained on-device
```

> See [`Applications/TransferLearning`](Applications/TransferLearning) — CIFAR classification & emotion recognition from hand-drawn images.

#### Few-Shot Learning
Learn new classes from as few as **1~5 examples** using centroid-based nearest-neighbor classification — no gradient updates needed at deployment time.

> See [`Applications/SimpleShot`](Applications/SimpleShot) — 73% accuracy with just 20 examples per class.

#### Full Model Training
Train entire CNNs, RNNs, and Transformers from scratch on device. NNTrainer's memory-optimized runtime makes it feasible even on resource-constrained hardware.

> See [`Applications/MNIST`](Applications/MNIST), [`Applications/Resnet`](Applications/Resnet), [`Applications/VGG`](Applications/VGG)

#### Reinforcement Learning
Complete Deep Q-Learning with experience replay and dual network architecture — tested on Galaxy S9.

> See [`Applications/ReinforcementLearning`](Applications/ReinforcementLearning)

### Training Infrastructure

| Component | Details |
|-----------|---------|
| **Optimizers** | SGD, Adam, AdamW |
| **LR Schedulers** | Constant, Exponential, Step, Cosine Annealing, Linear Decay |
| **Loss Functions** | Cross-Entropy (Softmax/Sigmoid), MSE, KL Divergence |
| **Regularization** | L2 Regularization, Dropout, Batch Normalization, Gradient Clipping |
| **Weight Init** | Xavier, He, LeCun (Normal/Uniform), Zeros |
| **Activations** | ReLU, GELU, Swish, Sigmoid, Tanh, Softmax, Mish, ELU, SELU, and more |
| **Data Loading** | File-based datasets or generator callbacks for streaming/augmentation |
| **Augmentation** | Random flip, translate, L2 normalization (built-in preprocessing layers) |
| **Export Formats** | Binary, INI, FlatBuffer, ONNX, TFLite |

### Why Train On-Device?

- **Privacy** — Sensitive data (health, biometrics, personal photos) stays on-device. No cloud upload required.
- **Personalization** — Adapt a generic model to each user's unique patterns and preferences in real-time.
- **Offline Capability** — Train and improve models without any network connectivity.
- **Low Latency** — No round-trip to the cloud. Instant feedback loop between data collection and model update.

---

## What's New

| Feature | Description |
|---------|-------------|
| **Qwen3 / Qwen3-MoE Support** | Full support for Qwen3 family including 30B MoE with on-device expert loading |
| **GPT-OSS 120B-5.1B** | Run 120B-parameter MoE models with cached-slim expert loading |
| **Gemma3 Support** | Google's Gemma3 architecture added to CausalLM |
| **GGML Quantizer** | Quantize models to reduced precision for smaller footprint and faster inference |
| **AVX2 GELU / Tanh-GELU** | SIMD-optimized activation kernels for x86_64 |
| **NEON SwiGLU / GELU** | ARM NEON SIMD optimizations with loop unrolling for mobile performance |
| **Android Benchmark Suite** | End-to-end benchmarking with thermal monitoring, device utilities, and tokenizer support |
| **Decoupled KV Cache** | Optimized attention for Qwen3 with separated query and KV cache paths |
| **MoE Expert Caching** | Cached-slim variants that keep hot experts in memory across inference steps |
| **Mixed Precision (FP16)** | Half-precision support for reduced memory and accelerated computation |
| **Windows ARM/x86_64** | Full build support for Windows platforms |

---

## Key Features

- **Run Locally, Fully Offline** — Training and inference on edge devices with zero cloud dependency. Data stays on the device.
- **On-Device Training & Personalization** — Fine-tune models on-device with private user data. Supports Transfer Learning, Few-Shot Learning, and Continuous Learning.
- **Efficient LLM Inference** — Run LLMs up to 120B parameters on memory-constrained devices with FSU and MoE caching.
- **Broad Model Support** — CNNs (ResNet, VGG, AlexNet, YOLO), RNNs (LSTM, GRU), Transformers (Qwen3, GPT-OSS, Gemma3, LLaMA), and Reinforcement Learning.
- **High Performance** — NEON/AVX2 SIMD, OpenCL GPU, cuBLAS, and NPU acceleration. Optimized memory pool and lazy tensor computation.
- **Cross-Platform** — Tizen, Android, Linux, Windows with consistent C/C++ APIs.

---

## Applications Gallery

NNTrainer provides **20+ ready-to-run example applications**:

| Category | Examples |
|----------|----------|
| **LLM / Transformers** | [CausalLM](Applications/CausalLM) 
| **Computer Vision** | [ResNet](Applications/Resnet), [VGG](Applications/VGG), [AlexNet](Applications/AlexNet), [YOLOv2](Applications/YOLOv2), [YOLOv3](Applications/YOLOv3), [MNIST](Applications/MNIST) |
| **Few-Shot / Transfer** | [SimpleShot](Applications/SimpleShot), [TransferLearning](Applications/TransferLearning) |
| **RL / Classical ML** | [ReinforcementLearning](Applications/ReinforcementLearning), [KNN](Applications/KNN), [LogisticRegression](Applications/LogisticRegression) |
| **Export / Interop** | [ONNX](Applications/ONNX), [TFLite Export](Applications/TFlite_export) |
| **Platform** | [Android (Kotlin/Java)](Applications/Android), [Tizen Native](Applications/Tizen_native) |

---

## Official Releases

|        | [Tizen](http://download.tizen.org/snapshots/tizen/unified/latest/repos/standard/packages/) | [Ubuntu](https://launchpad.net/~nnstreamer/+archive/ubuntu/ppa) | Android/NDK | Windows |
| :----- | :--: | :--: | :--: | :--: |
|        | 7.0M2+ | 22.04 / 24.04 | 9/P | 2022+ |
| arm64  | [![Tizen ARM](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_tizen_arm.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_tizen_arm.yml) | [![Ubuntu](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build.yml) | [![Android](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_android.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_android.yml) | [![Windows ARM](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_windows_arm.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_windows_arm.yml) |
| x86_64 | [![Tizen x86_64](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_tizen_x86_64.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_tizen_x86_64.yml) | [![Ubuntu](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build.yml) | N/A | [![Windows x86_64](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_windows_x86_64.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_windows_x86_64.yml) |
| API    | C (Official) | C/C++ | C/C++ | C/C++ |

- **SDK Support**: Tizen Studio 7.0+
- **Binary Packages**: Tizen Repo, Ubuntu PPA

---

## Getting Started

- **[Installation Guide](https://github.com/nntrainer/nntrainer/blob/main/docs/getting-started.md)** — Build & install on Linux, Android, or Windows
- **[Create Your Model](https://github.com/nntrainer/nntrainer/blob/main/docs/how-to-create-model.md)** — Tutorial for building custom models
- **[Run Examples](https://github.com/nntrainer/nntrainer/blob/main/docs/how-to-run-examples.md)** — Step-by-step guide for running applications
- **[Supported Components](https://github.com/nntrainer/nntrainer/blob/main/docs/components.md)** — Full list of layers, optimizers, loss functions, and activations
- **[C API Reference](https://github.com/nntrainer/nntrainer/blob/master/api/capi/include/nntrainer.h)** | **[C++ API Reference](https://github.com/nntrainer/nntrainer/blob/master/api/ccapi/include)**

---

## Publications

- [Memory-Efficient LLM Inference on Edge Devices With NNTrainer](https://youtu.be/J2tUmi4bwMY?si=rJyiXkwr5iFrMhIK) — Open Source Summit 2025 Seoul
- [A New Frontier of AI: On-Device AI Training and Personalization](https://dl.acm.org/doi/abs/10.1145/3639477.3639716) — ICSE-SEIP, 2024
- [NNTrainer: Light-Weight On-Device Training Framework](https://arxiv.org/pdf/2206.04688.pdf) — arXiv, 2022
- [Open Source On-Device AI SW Platform](https://youtu.be/im3uNrPLYx4?si=gMbw7LKKSnpXi59U) — Samsung Developer Conference 2023
- [NNTrainer: Personalize neural networks on devices!](https://www.youtube.com/watch?v=HKKowY78P1A) — Samsung Developer Conference 2021
- [NNTrainer: "On-device learning"](https://www.youtube.com/embed/Jy_auavraKg?start=4035&end=4080) — Samsung AI Forum 2021

---

## Contributing

Contributions are welcome! Please see our [Contributing Guide](https://github.com/nntrainer/nntrainer/blob/main/docs/contributing.md).

## License

Apache License 2.0

## Citation

If you find NNTrainer useful, please cite our paper:

```bibtex
@inproceedings{10.1145/3639477.3639716,
  author = {Moon, Jijoong and Lee, Hyeonseok and Chu, Jiho and Park, Donghak and Hong, Seungbaek and Seo, Hyungjun and Jeong, Donghyeon and Kong, Sungsik and Ham, Myungjoo},
  title = {A New Frontier of AI: On-Device AI Training and Personalization},
  year = {2024},
  isbn = {9798400705014},
  publisher = {Association for Computing Machinery},
  url = {https://doi.org/10.1145/3639477.3639716},
  doi = {10.1145/3639477.3639716},
  booktitle = {Proceedings of the 46th International Conference on Software Engineering: Software Engineering in Practice},
  pages = {323--333},
  numpages = {11},
  keywords = {on-device AI, neural network, personalization, training, software framework},
  series = {ICSE-SEIP '24}
}
```
