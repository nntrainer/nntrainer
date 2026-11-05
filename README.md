<p align="center">
  <h1 align="center">NNTrainer</h1>
  <p align="center">
    <strong>On-device AI framework</strong><br/>
    Run LLMs on a phone's CPU or NPU. Train models where the data is.
  </p>
</p>

<p align="center">
  <a href="https://github.com/nntrainer/nntrainer/actions/workflows/daily_build.yml"><img src="https://github.com/nntrainer/nntrainer/actions/workflows/daily_build.yml/badge.svg" alt="Daily build"/></a>
  <a href="https://nntrainer.github.io/coverage_result/"><img src="https://img.shields.io/endpoint?url=https://nntrainer.github.io/coverage_result/coverage.json" alt="Code coverage"/></a>
  <a href="https://scan.coverity.com/projects/nnstreamer-nntrainer"><img src="https://scan.coverity.com/projects/22512/badge.svg" alt="Coverity scan"/></a>
  <a href="https://www.bestpractices.dev/projects/9179"><img src="https://www.bestpractices.dev/projects/9179/badge" alt="OpenSSF best practices"/></a>
</p>

NNTrainer is a C++ framework for neural networks on devices with little
memory: phones, TVs, appliances.

- **Run LLMs on a phone.** Mixture-of-experts models of 20 to 30 billion
  parameters, on the CPU or on the Hexagon NPU.
- **Train on the device.** Fine-tune and personalize models without sending
  data anywhere.
- **Embed it.** C++ and C APIs, the QuickDotAI Android library and the
  Quick.AI app.

## Try it

| | Path | You need |
|--:|---|---|
| 1 | [An LLM on your PC](#1-an-llm-on-your-pc) | Ubuntu, `meson`, a C++17 compiler |
| 2 | [An LLM on an Android phone](#2-an-llm-on-an-android-phone) | Android NDK, a phone |
| 3 | [Gemma4 26B or LFM2 24B on the NPU](#3-gemma4-26b-or-lfm2-24b-on-the-npu) | Galaxy S26 Ultra, Hexagon SDK, HexKL, a model pack |
| 4 | [Train a model](#4-train-a-model) | the host build |
| 5 | [Use it from your program](#5-use-it-from-your-program) | C++, C, Kotlin or Python |

### 1. An LLM on your PC

Build with the transformer option ([prerequisites](docs/getting-started.md#build-with-meson)):

```bash
git clone https://github.com/nntrainer/nntrainer.git && cd nntrainer
git submodule sync && git submodule update --init --depth 1
meson setup build -Denable-transformer=true && ninja -C build
```

Prepare a small model. Put `config.json`, `generation_config.json`,
`tokenizer.json` and `tokenizer_config.json` from the Qwen3-0.6B Hugging
Face repository in a directory, then convert the GGUF file into it:

```bash
cd ~/models/qwen3-0.6b
python3 ~/nntrainer/Applications/CausalLM/res/qwen3/qwen3-0.6b/gguf_to_nntrainer.py \
  /path/to/qwen3-0.6b.gguf -o nntr_qwen3_0.6b_q40_embdq6k.bin --target x86 --emit-nntr-config
```

Ask it something:

```bash
NNTR_NUM_THREADS=4 ~/nntrainer/build/Applications/CausalLM/nntr_causallm ~/models/qwen3-0.6b "What is on-device AI?"
```

The tool prints the answer with its prefill and generation speeds.
Next: [more models, quantization, the GPU path](Applications/CausalLM/README.md).

### 2. An LLM on an Android phone

Build for arm64 and push the tool to the phone:

```bash
export ANDROID_NDK=/path/to/android-ndk
cd Applications/CausalLM && ./build_android.sh --install
```

Convert the model as in path 1 with `--target arm`, push it, run it:

```bash
adb push ~/models/qwen3-0.6b-arm /sdcard/Download/qwen3-0.6b
adb shell sh /data/local/tmp/Quick.AI/run_causallm.sh /sdcard/Download/qwen3-0.6b "What is on-device AI?"
```

Next: [Android build options](Applications/CausalLM/README.md#4-android-build--test).

### 3. Gemma4 26B or LFM2 24B on the NPU

Build the DSP image, the libraries and the Quick.AI app, and install them:

```bash
source /path/to/Hexagon_SDK/6.4.0.1/setup_sdk_env.source
export HEXKL_ROOT=/path/to/hexkl_addon HEX_ARCH=v81 ANDROID_NDK=/path/to/android-ndk-r30
cd Applications/CausalLM && ./build_android.sh --htp --app --install
```

Put the model pack on the phone ([where packs come from](docs/quickai/model-packs.md#where-packs-come-from)):

```bash
adb push gemma4-26b-a4b-htp-w4a32 /sdcard/Download/aistudio-mobile/models/
```

Open **Quick.AI**, tap the model chip, pick `gemma4-26b-a4b` · `Quick.AI` ·
`NPU`, tap **Load model**, ask a question. Decode runs at 12 to 19 tokens
per second on the S26 Ultra.
Next: [the Quick.AI documentation](docs/quickai/README.md): app, library, C
API, packs, command line, configuration, performance.

### 4. Train a model

The training runtime and the examples build with the default options:

```bash
meson setup build && ninja -C build && cd build
export MNIST_RES=$PWD/res/app/MNIST
./Applications/MNIST/jni/nntrainer_mnist $MNIST_RES/mnist.ini $MNIST_RES/mnist_trainingSet.dat
```

The same model from C++ ([full program](docs/how-to-create-model.md)):

```cpp
auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});
model->addLayer(ml::train::createLayer("input", {"input_shape=1:1:10"}));
model->addLayer(ml::train::createLayer("fully_connected", {"unit=5", "activation=softmax"}));
model->setProperty({"batch_size=4", "epochs=10", "save_path=my_model.bin"});
model->setOptimizer(ml::train::createOptimizer("SGD", {"learning_rate=0.001"}));
model->compile();
model->initialize();
model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, std::move(dataset));
model->train();
```

A model can also be an INI file, no code
([example](Applications/MNIST/res/mnist.ini), [reference](docs/configuration-ini.md)).
Next: transfer learning, few-shot and reinforcement learning examples in
[`Applications/`](Applications/README.md).

### 5. Use it from your program

- **C++**: the `ml::train` API (`model.h`, `layer.h`, `optimizer.h`,
  `dataset.h`) and the tensor graph API `tensor_api.h`. Start at
  [api/ccapi](api/ccapi/README.md).
- **C**: `nntrainer.h`, Tizen's ML training API.
  [api/capi/include/nntrainer.h](api/capi/include/nntrainer.h).
- **Kotlin or Java, LLM**: the QuickDotAI AAR. Load a model by catalog id,
  stream the answer. [QuickDotAI library](Applications/CausalLM/Android/QuickDotAI/README.md).
- **C or C++, LLM**: `quick_dot_ai_api.h`, the handle-based API under the
  AAR. [QuickDotAI C API](Applications/CausalLM/api/README.md).
- **Python, model preparation**: the converters under
  `Applications/CausalLM/res/<model>/` and `nntr_quantize`.
  [Quantizing models](Applications/CausalLM/README.md#quantizing-models).

## What runs where

| Model | Sizes | Runs on |
|---|---|---|
| Qwen3 | 0.6B to 32B | CPU, GPU |
| Qwen3-MoE | 30B-A3B | CPU, experts streamed from flash |
| GPT-OSS | 20B-A3.6B, 120B-5.1B | CPU, experts streamed from flash |
| Gemma2, Gemma3 | 2B and up | CPU; Gemma2 also GPU |
| Gemma4 E2B | 2B | CPU, GPU, QNN NPU; LiteRT-LM in the app |
| **Gemma4** | **26B-A4B** | **Hexagon NPU** |
| **LFM2** | **24B-A2B** | **Hexagon NPU** |
| LLaMA, Qwen2 | | CPU |
| BERT, XLM-RoBERTa, DeBERTa-v2, sentence embeddings, timm ViT, V-JEPA 2 | | CPU; V-JEPA 2 also QNN NPU |

GPU means OpenCL (Adreno, Intel Arc). Training runs on the CPU: CNNs, RNNs
(LSTM, GRU), transformers and reinforcement learning, with SGD, Adam and
AdamW ([all components](docs/components.md)).

## Documentation

- **Install and build**: [Linux and Tizen](docs/getting-started.md),
  [Windows](docs/getting-started-windows.md)
- **LLM engine**: [Applications/CausalLM](Applications/CausalLM/README.md),
  [GPU guide](Applications/CausalLM/README_GPU.md)
- **Quick.AI on the Hexagon NPU**: [docs/quickai](docs/quickai/README.md)
- **Build a model**: [in C++](docs/how-to-create-model.md),
  [in INI](docs/configuration-ini.md)
- **Run the examples**: [on Linux](docs/how-to-run-examples.md),
  [on Android](docs/how-to-run-example-android.md)
- **Components**: [layers, optimizers, losses, activations](docs/components.md)
- **Internals**: [backend dispatch](docs/backend_guide/ARCHITECTURE.md),
  [the HTP backend](docs/backend_guide/HTP_ARCHITECTURE.md),
  [environment variables](docs/ENV_FLAGS.md)
- **Contributing**: [guide](docs/contributing.md),
  [coding convention](docs/coding-convention.md)

## Platforms

- **Ubuntu 22.04 / 24.04**
  [![Ubuntu](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build.yml)
  Packages: [PPA `nnstreamer/ppa`](https://launchpad.net/~nnstreamer/+archive/ubuntu/ppa)
- **Android 10 and later** (API 29; the Quick.AI app needs Android 13)
  [![Android](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_android.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_android.yml)
- **Tizen 7.0 M2 and later**
  [![Tizen ARM](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_tizen_arm.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_tizen_arm.yml)
  [![Tizen x86_64](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_tizen_x86_64.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_tizen_x86_64.yml)
  Packages: [Tizen repository](http://download.tizen.org/snapshots/tizen/unified/latest/repos/standard/packages/)
- **Windows 11**
  [![Windows ARM](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_windows_arm.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_windows_arm.yml)
  [![Windows x86_64](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_windows_x86_64.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_windows_x86_64.yml)

## Publications

- [Memory-Efficient LLM Inference on Edge Devices With NNTrainer](https://youtu.be/J2tUmi4bwMY?si=rJyiXkwr5iFrMhIK), Open Source Summit Seoul 2025
- [A New Frontier of AI: On-Device AI Training and Personalization](https://dl.acm.org/doi/abs/10.1145/3639477.3639716), ICSE-SEIP 2024
- [NNTrainer: Light-Weight On-Device Training Framework](https://arxiv.org/pdf/2206.04688.pdf), arXiv 2022

## License

Apache License 2.0. Contributions are welcome: see the
[contributing guide](docs/contributing.md).

<details>
<summary>Cite NNTrainer</summary>

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

</details>
