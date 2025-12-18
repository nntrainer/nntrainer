# NNtrainer

[![Code Coverage](https://img.shields.io/endpoint?url=https://nntrainer.github.io/coverage_result/coverage.json)](https://nntrainer.github.io/coverage_result/)
[![DailyBuild](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build.yml/badge.svg)](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build.yml)
![GitHub repo size](https://img.shields.io/github/repo-size/nnstreamer/nntrainer)
![GitHub issues](https://img.shields.io/github/issues/nnstreamer/nntrainer)
![GitHub pull requests](https://img.shields.io/github/issues-pr/nnstreamer/nntrainer)
<a href="https://scan.coverity.com/projects/nnstreamer-nntrainer">
  <img alt="Coverity Scan Build Status"
       src="https://scan.coverity.com/projects/22512/badge.svg"/>
</a>
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9179/badge)](https://www.bestpractices.dev/projects/9179)

NNtrainer is a Software Framework for training/Inference of Neural Network models on devices.

## Overview

NNtrainer is an Open Source Project. The aim of the NNtrainer is to develop a Software Framework to train and run inference of neural network models on embedded devices which have relatively limited resources.

**Training**: Rather than training whole layers of a network from the scratch, NNtrainer finetunes the neural network model on device with user data for the personalization.

**Inference**: NNtrainer supports efficient LLM inference on memory-constrained devices through advanced memory optimization techniques:
- **FSU (Flash Storage Utilization)**: Dynamically loads model weights from flash storage during inference, significantly reducing peak memory usage.
- **MoE Cache**: Intelligent caching mechanism for Mixture of Experts models, keeping frequently used experts in memory while swapping others to storage.
- **Proactive Loading**: Predicts and pre-loads required weights before they are needed, minimizing latency during inference.

Even if NNtrainer runs on device, it provides full functionalities to train and infer models and also utilizes limited device resources efficiently. NNTrainer supports various machine learning algorithms such as k-Nearest Neighbor (k-NN), Neural Networks, Logistic Regression, Reinforcement Learning algorithms, Recurrent network, Transformers (LLM) and more. We also provide examples for various tasks such as Few-shot learning, ResNet, VGG, Product Rating, LLM Inference and more will be added. All of these were tested on Samsung Galaxy smart phone with Android and PC (Ubuntu).

[ Memory-Efficient LLM Inference on Edge Devices With NNTrainer ](https://youtu.be/J2tUmi4bwMY?si=rJyiXkwr5iFrMhIK), Open Source Summit 2025 Seoul <br />
[ A New Frontier of AI: On-Device AI Training and Personalization ](https://dl.acm.org/doi/abs/10.1145/3639477.3639716), ICSE-SEIP, 2024 <br />
[ NNTrainer: Light-Weight On-Device Training Framework ](https://arxiv.org/pdf/2206.04688.pdf), arXiv, 2022 <br />
[  Open Source On-Device AI SW Platform ](https://youtu.be/im3uNrPLYx4?si=gMbw7LKKSnpXi59U), Samsung Developer Conference 2023 (Korean) <br />
[ NNTrainer: Personalize neural networks on devices! ](https://www.youtube.com/watch?v=HKKowY78P1A), Samsung Developer Conference 2021 <br />
[ NNTrainer: "On-device learning" ](https://www.youtube.com/embed/Jy_auavraKg?start=4035&end=4080), Samsung AI Forum 2021

## ✨ Key Features

- 📱 **Run Locally, Fully Offline**: Perform training and inference directly on your edge devices without any internet connection. Ensure data privacy and low latency.
- 🛠️ **On-Device Training & Personalization**: Fine-tune models on-device with private user data. Supports Transfer Learning, Few-Shot Learning, and Continuous Learning scenarios.
- 🚀 **Efficient LLM Inference**: Run Large Language Models on memory-constrained devices. Features **Flash Storage Utilization (FSU)** to offload weights and **MoE Cache** for efficient Mixture-of-Experts execution.
- 🤖 **Broad Model Support**: Built-in support for various architectures including CNNs (ResNet, VGG), RNNs (LSTM, GRU), Transformers (LLaMA, Qwen, DeepSeek), and Reinforcement Learning.
- ⚡ **High Performance & Lightweight**: Optimized for embedded systems. Functionalities are designed to minimize memory footprint and maximize execution speed on ARM, x86, and NPU targets.
- 🔌 **Cross-Platform**: Deploy seamlessly across Tizen, Android, Linux, and Windows environments with consistent C/C++ APIs.


## 🚀 Running LLMs On Device

NNTrainer supports Large Language Model (LLM) inference! Please refer to `Applications/CausalLM` for details. We currently support various LLM architectures, including:
  - Qwen3
  - Qwen3-MoE
  - GPT-OSS
  - And more to come!

### Running MoE on your device
  NNTrainer enables the execution of large-scale Mixture of Experts (MoE) models directly on-device. We support on-the-fly expert loading using flash storage, allowing MoE models to run with significantly reduced peak memory usage.
  Below are demonstrations of MoE LLMs running on mobile devices:

#### 📱 Running MoE on your mobile phone

| GPT-OSS 20B | Qwen3 MoE 30B-A3B |
|:-----------:|:-----------------:|
| <img src="docs/videos/GPT_OSS_20B_Demo.gif" width="360"> | <img src="docs/videos/Qwen_30B_Demo.gif" width="360"> |

#### 💻 Running MoE on your PC

 NNTrainer's FSU feature allows for dynamic expert loading during inference. This minimizes initialization time and ensures efficiency in memory-constrained environments. To test this, try the models located in `Applications/CausalLM/models/*-slim`.

  | Load Whole model (Qwen3-30B-A3B) | Load Expert On-The-Fly (Qwen3-30B-A3B-Slim)|
  |:-------:|:------:|
  |![](./docs/videos/moe-full.gif)| ![](./docs/videos/moe-on-the-fly.gif) |
  | Memory: 16.5 GB | Memory: 1.3 GB |


## Official Releases

|     | [Tizen](http://download.tizen.org/snapshots/tizen/unified/latest/repos/standard/packages/) | [Ubuntu](https://launchpad.net/~nnstreamer/+archive/ubuntu/ppa) |                                                                                      Android/NDK Build                                                                                       |                                                                                                      Windows                                                                                                      |
| :-- | :--: | :--: |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     | 7.0M2 and later | 22.04/24.04 |                                                                                             9/P                                                                                              |                                                                                              windows-2022 and later                                                                                               |
| arm64 | [![Tizen ARM](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build_tizen_arm.yml/badge.svg)](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build_tizen_arm.yml) | [![Ubuntu](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build.yml/badge.svg)](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build.yml) | [![Android](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build_android.yml/badge.svg)](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build_android.yml) |     [![Windows ARM](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build_windows_arm.yml/badge.svg)](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build_windows_arm.yml)      |
| x86_64 | [![Tizen x86_64](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build_tizen_x86_64.yml/badge.svg)](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build_tizen_x86_64.yml) | [![Ubuntu](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build.yml/badge.svg)](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build.yml) |                                                                                             N/A                                                                                              | [![Windows x86_64](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build_windows_x86_64.yml/badge.svg)](https://github.com/nnstreamer/nntrainer/actions/workflows/daily_build_windows_x86_64.yml) |
| Publish | [Tizen Repo](http://download.tizen.org/snapshots/tizen/unified/latest/repos/standard/packages/) | [PPA](https://launchpad.net/~nnstreamer/+archive/ubuntu/ppa) |                                                                                              NA                                                                                              |                                                                                                        NA                                                                                                         |
| API | C (Official) | C/C++ |                                                                                            C/C++                                                                                             |                                                                                                       C/C++                                                                                                       |

- Ready: CI system ensures build-ability and unit-testing. Users may easily build and execute. However, we do not have automated release & deployment system for this instance.
- Available: binary packages are released and deployed automatically and periodically along with CI tests.
- SDK Support: Tizen Studio (7.0 and later)

## Getting Started

### [Installation](https://github.com/nnstreamer/nntrainer/blob/main/docs/getting-started.md)

Instructions for installing NNTrainer.

### [Tutorial](https://github.com/nnstreamer/nntrainer/blob/main/docs/how-to-create-model.md)

Introductions for creating your own model.

### [Running Examples](https://github.com/nnstreamer/nntrainer/blob/main/docs/how-to-run-examples.md)

Instructions for preparing NNTrainer for execution

### [Examples for NNTrainer](https://github.com/nnstreamer/nntrainer/tree/main/Applications)

NNTrainer examples for a variety of networks

### [Components](https://github.com/nnstreamer/nntrainer/blob/main/docs/components.md)

[Our supported features](https://github.com/nnstreamer/nntrainer/blob/main/docs/components.md#supported_feature)
- Supported ptimizer
- Supported Loss Functions
- Supported Activation Functions
- Supported Tensor
- Others

### APIs
Currently, we provide [C APIs](https://github.com/nnstreamer/nntrainer/blob/master/api/capi/include/nntrainer.h) for Tizen. [C++ APIs](https://github.com/nnstreamer/nntrainer/blob/master/api/ccapi/include) are also provided for other platform. Java & C# APIs will be provided soon.

## Open Source License

The NNtrainer is an open source project released under the terms of the Apache License version 2.0.

## Contributing

Contributions are welcome! Please see our [Contributing](https://github.com/nnstreamer/nntrainer/blob/main/docs/contributing.md) Guide for more details.

[//]: # ([![]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/0&#41;]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/0&#41;[![]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/1&#41;]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/1&#41;[![]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/2&#41;]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/2&#41;[![]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/3&#41;]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/3&#41;[![]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/4&#41;]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/4&#41;[![]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/5&#41;]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/5&#41;[![]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/6&#41;]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/6&#41;[![]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/images/7&#41;]&#40;https://sourcerer.io/fame/dongju-chae/nnstreamer/nntrainer/links/7&#41;)

## Citation

If you find this NNTrainer project useful or relevant to your research, please consider citing our paper:


```
@inproceedings{10.1145/3639477.3639716,
author = {Moon, Jijoong and Lee, Hyeonseok and Chu, Jiho and Park, Donghak and Hong, Seungbaek and Seo, Hyungjun and Jeong, Donghyeon and Kong, Sungsik and Ham, Myungjoo},
title = {A New Frontier of AI: On-Device AI Training and Personalization},
year = {2024},
isbn = {9798400705014},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3639477.3639716},
doi = {10.1145/3639477.3639716},
booktitle = {Proceedings of the 46th International Conference on Software Engineering: Software Engineering in Practice},
pages = {323–333},
numpages = {11},
keywords = {on-device AI, neural network, personalization, training, software framework},
series = {ICSE-SEIP '24}
}
```
