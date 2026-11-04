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

NNTrainer is a C++ framework for running and training neural networks on
devices with limited memory: phones, TVs, appliances. It ships an LLM
inference engine (`Applications/CausalLM`) that runs mixture-of-experts
models of 20 to 30 billion parameters on a phone, on the CPU or on the
Hexagon NPU, and a training runtime that fine-tunes models on the device
without sending data anywhere. Apache License 2.0.

## Try it

Pick the path that matches what you have. Each one is a complete, copyable
sequence; the linked page has the details and the troubleshooting.

### Run an LLM on your PC

Needs Ubuntu 22.04 or 24.04 with `meson`, `ninja`, a C++17 compiler and
Python 3 ([build prerequisites](docs/getting-started.md#build-with-meson)).

```bash
git clone https://github.com/nntrainer/nntrainer.git && cd nntrainer
git submodule sync && git submodule update --init --depth 1
meson setup build -Denable-transformer=true
ninja -C build

# A small model: Qwen3-0.6B from a GGUF file, with the config and tokenizer
# JSON files of the Hugging Face repository copied next to it.
mkdir -p ~/models/qwen3-0.6b && cd ~/models/qwen3-0.6b
python3 ~/nntrainer/Applications/CausalLM/res/qwen3/qwen3-0.6b/gguf_to_nntrainer.py \
  /path/to/qwen3-0.6b.gguf -o nntr_qwen3_0.6b_q40_embdq6k.bin --target x86 --emit-nntr-config

NNTR_NUM_THREADS=4 ~/nntrainer/build/Applications/CausalLM/nntr_causallm ~/models/qwen3-0.6b "What is on-device AI?"
```

The tool prints the answer with its prefill and generation speeds.
[More models, quantization and the GPU path](Applications/CausalLM/README.md).

### Run an LLM on an Android phone

Needs the Android NDK and a phone with USB debugging on.

```bash
export ANDROID_NDK=/path/to/android-ndk
cd Applications/CausalLM
./build_android.sh --install          # builds for arm64 and pushes the tool to /data/local/tmp/Quick.AI

# convert the model for ARM (--target arm) and push it
adb push ~/models/qwen3-0.6b-arm /sdcard/Download/qwen3-0.6b
adb shell sh /data/local/tmp/Quick.AI/run_causallm.sh /sdcard/Download/qwen3-0.6b "What is on-device AI?"
```

[Android build options, the QuickDotAI library and the sample app](Applications/CausalLM/README.md#4-android-build--test).

### Run Gemma4 26B or LFM2 24B on the Hexagon NPU

Needs a Galaxy S26 Ultra (Hexagon v81), the Qualcomm Hexagon SDK 6.4 and
HexKL, and a model pack ([where packs come from](docs/quickai/model-packs.md#where-packs-come-from)).

```bash
source /path/to/Hexagon_SDK/6.4.0.1/setup_sdk_env.source
export HEXKL_ROOT=/path/to/hexkl_addon HEX_ARCH=v81 ANDROID_NDK=/path/to/android-ndk-r30
cd Applications/CausalLM
./build_android.sh --htp --app --install      # DSP image, libraries, Quick.AI app, all on the phone

adb push gemma4-26b-a4b-htp-w4a32 /sdcard/Download/aistudio-mobile/models/
```

Open **Quick.AI** on the phone, tap the model chip, choose
`gemma4-26b-a4b` · `Quick.AI` · `NPU`, tap **Load model** and ask a
question. Decode runs at 12 to 19 tokens per second on the S26 Ultra.
[The Quick.AI documentation](docs/quickai/README.md) covers the app, the
Android library, the C API, packs, the command line and the configuration.

### Train a model on device

The training runtime builds with the default options; the examples under
`Applications/` build with it.

```bash
meson setup build
ninja -C build
cd build
export MNIST_RES=$PWD/res/app/MNIST
./Applications/MNIST/jni/nntrainer_mnist $MNIST_RES/mnist.ini $MNIST_RES/mnist_trainingSet.dat
```

The same thing from C++ ([full tutorial](docs/how-to-create-model.md)):

```cpp
#include <layer.h>
#include <model.h>
#include <optimizer.h>

auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});
model->addLayer(ml::train::createLayer("input", {"input_shape=1:1:10"}));
model->addLayer(ml::train::createLayer("fully_connected", {"unit=5", "activation=softmax"}));
model->setProperty({"batch_size=4", "epochs=10", "save_path=my_model.bin"});
model->setOptimizer(ml::train::createOptimizer("SGD", {"learning_rate=0.001"}));
model->compile();
model->initialize();
model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, std::move(dataset)); // a file or a generator callback
model->train();
```

Models can also be described in an INI file (`Applications/MNIST/res/mnist.ini`)
and loaded without code. Transfer learning, few-shot learning and
reinforcement learning examples are in
[`Applications/`](Applications/README.md).

### Use NNTrainer from your own program

| You write | Use | Start at |
|---|---|---|
| C++ | `ml::train` API: `model.h`, `layer.h`, `optimizer.h`, `dataset.h`, and the tensor graph API `tensor_api.h` | [api/ccapi](api/ccapi/README.md) |
| C | `nntrainer.h` (Tizen's official ML training API) | [api/capi/include/nntrainer.h](api/capi/include/nntrainer.h) |
| Kotlin / Java (LLM) | the QuickDotAI AAR: load a model by catalog id, stream the answer | [QuickDotAI library](Applications/CausalLM/Android/QuickDotAI/README.md) |
| C / C++ (LLM) | `quick_dot_ai_api.h`: the handle-based LLM API under the AAR | [QuickDotAI C API](Applications/CausalLM/api/README.md) |
| Python (model prep) | the converters under `Applications/CausalLM/res/<model>/` and `nntr_quantize` | [CausalLM README](Applications/CausalLM/README.md#quantizing-models) |

## What runs where

| Model | Sizes | CPU | GPU (OpenCL) | NPU | Notes |
|---|---|:-:|:-:|:-:|---|
| Qwen3 | 0.6B – 32B | ✓ | ✓ | | Q4_0 / QS4CX weights, FP16 or FP32 activations |
| Qwen3-MoE | 30B-A3B | ✓ | | | experts loaded from flash on demand (`slim`), with caching (`cached_slim`) |
| GPT-OSS | 20B-A3.6B, 120B-5.1B | ✓ | | | `cached_slim` expert loading |
| Gemma2, Gemma3 | 2B and up | ✓ | ✓ (Gemma2) | | |
| Gemma4 E2B | 2B | ✓ | ✓ | ✓ QNN | also through LiteRT-LM in the Quick.AI app |
| **Gemma4** | **26B-A4B** | | | **✓ Hexagon HTP** | 2-bit or ternary experts, int4 projections, int8 KV cache |
| **LFM2** | **24B-A2B** | | | **✓ Hexagon HTP** | ternary experts, affine int4 projections |
| LLaMA, Qwen2 | | ✓ | | | |
| Encoders and vision | BERT, XLM-RoBERTa, DeBERTa-v2, sentence embeddings, timm ViT, V-JEPA 2 | ✓ | | ✓ QNN (V-JEPA 2) | |

Training: CNNs, RNNs (LSTM, GRU), transformers and reinforcement learning on
the CPU; optimizers SGD, Adam, AdamW with constant, exponential, step,
cosine and linear schedules; see [supported components](docs/components.md).

## Documentation

| | |
|---|---|
| Build and install on Linux, Tizen, Windows | [docs/getting-started.md](docs/getting-started.md), [Windows](docs/getting-started-windows.md) |
| LLM inference engine: models, quantization, PC and Android builds, GPU | [Applications/CausalLM/README.md](Applications/CausalLM/README.md), [README_GPU.md](Applications/CausalLM/README_GPU.md) |
| Quick.AI on the Hexagon NPU: app, library, C API, packs, command line, configuration, performance | [docs/quickai/README.md](docs/quickai/README.md) |
| Building a model in C++ or INI | [docs/how-to-create-model.md](docs/how-to-create-model.md), [docs/configuration-ini.md](docs/configuration-ini.md) |
| Running the example applications | [docs/how-to-run-examples.md](docs/how-to-run-examples.md), [on Android](docs/how-to-run-example-android.md) |
| Layers, optimizers, losses, activations | [docs/components.md](docs/components.md) |
| Backend internals: dispatch, vendor backends, the HTP backend | [docs/backend_guide/ARCHITECTURE.md](docs/backend_guide/ARCHITECTURE.md), [HTP_ARCHITECTURE.md](docs/backend_guide/HTP_ARCHITECTURE.md) |
| Runtime environment variables | [docs/ENV_FLAGS.md](docs/ENV_FLAGS.md) |
| Contributing | [docs/contributing.md](docs/contributing.md), [coding convention](docs/coding-convention.md) |

## Builds and packages

| | Tizen | Ubuntu | Android | Windows |
|:--|:--:|:--:|:--:|:--:|
| Versions | 7.0 M2 and later | 22.04 / 24.04 | 10 and later (API 29); the Quick.AI app needs 13 | Windows 11 |
| arm64 | [![Tizen ARM](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_tizen_arm.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_tizen_arm.yml) | [![Ubuntu](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build.yml) | [![Android](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_android.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_android.yml) | [![Windows ARM](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_windows_arm.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_windows_arm.yml) |
| x86_64 | [![Tizen x86_64](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_tizen_x86_64.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_tizen_x86_64.yml) | [![Ubuntu](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build.yml) | – | [![Windows x86_64](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_windows_x86_64.yml/badge.svg)](https://github.com/nntrainer/nntrainer/actions/workflows/daily_build_windows_x86_64.yml) |
| Packages | [Tizen repository](http://download.tizen.org/snapshots/tizen/unified/latest/repos/standard/packages/) | [PPA `nnstreamer/ppa`](https://launchpad.net/~nnstreamer/+archive/ubuntu/ppa) | build from source | build from source |

## Publications

- [Memory-Efficient LLM Inference on Edge Devices With NNTrainer](https://youtu.be/J2tUmi4bwMY?si=rJyiXkwr5iFrMhIK), Open Source Summit Seoul 2025
- [A New Frontier of AI: On-Device AI Training and Personalization](https://dl.acm.org/doi/abs/10.1145/3639477.3639716), ICSE-SEIP 2024
- [NNTrainer: Light-Weight On-Device Training Framework](https://arxiv.org/pdf/2206.04688.pdf), arXiv 2022

## Contributing and license

Contributions are welcome; see the [contributing guide](docs/contributing.md)
(DCO sign-off, `clang-format`). NNTrainer is released under the Apache
License 2.0.

If NNTrainer is useful in your research, please cite:

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
