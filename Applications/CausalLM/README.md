# ☄️ CausalLM Inference with NNTrainer

This application provides a standalone executable and an optional C API to run causal LLM models using NNTrainer.
It supports *inference* mode (text generation) on various devices, including Android.

## Features

- **Standalone Application (`nntr_causallm`)**: A command-line tool to load models and generate text.
- **C API (Optional)**: A lightweight C interface (`libcausallm_api.so`) for integrating LLM capabilities into other applications (e.g., Android JNI, iOS, or other C/C++ apps).
- **Core Library**: The core implementation is separated into `libcausallm_core.so` for modularity.
- **Supported Backends**: CPU, with GPU/NPU support planned.

## Supported models

- Llama
- Qwen3 (0.6B, 1.7B, 4B, 7B, 14B, 32B) [[link](https://huggingface.co/Qwen/Qwen3-4B)]
- Qwen3-MoE (30B-A3B) [[link](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)]
- GPT-OSS (MoE: 20B, 120B) [[link](https://huggingface.co/openai/gpt-oss-20b)]
- You can try your own model with custom layers!
- Feel free to contribute! 😊

For more details, please refer to the [Model Documentation](models/README.md).

## CausalLM API

The CausalLM application exposes a C API for easy integration with other applications (e.g., Android JNI).
The API allows loading models, running inference, and retrieving performance metrics.

For detailed documentation, please refer to [API Documentation](api/README.md).

## How to run

### 1. Prepare Model Files
- Download and copy the model files from huggingface to `res/{model}` directory.
- The folder should contain:
    - `config.json`
    - `generation_config.json`
    - `tokenizer.json`
    - `tokenizer_config.json`
    - `vocab.json`
    - `nntr_config.json`
    - nntrainer weight binfile (matches with the name in `nntr_config.json`)

### 2. PC Build & Test

Compile the application with transformer support enabled.

```bash
$ meson build -Denable-fp16=true -Dthread-backend=omp -Denable-transformer=true
$ ninja -C build
```

Run the model:

```bash
$ export NNTR_NUM_THREADS=4
$ ./build/Applications/CausalLM/nntr_causallm {your model config folder}
```

e.g.,
```bash
$ ./build/Applications/CausalLM/nntr_causallm /tmp/nntrainer/Applications/CausalLM/res/qwen3/qwen3-4b/
```

### 3. Android Build & Test

The Android build process is modularized to support building the core library, API library, and test applications independently.

#### Prerequisites
- Android NDK (e.g., r21d or later)
- CMake
- Rust (for tokenizers-cpp)
- ADB (Android Debug Bridge)

#### Build Scripts

The following scripts are provided in `Applications/CausalLM/` to handle the build process:

1.  **`build_android.sh`** (Core + App):
    - Builds `nntrainer` core library for Android.
    - Builds `tokenizers-cpp` dependency if missing.
    - Compiles **`libcausallm_core.so`** (Core logic) and **`nntrainer_causallm`** (Main Executable).
    - **Usage**: `./build_android.sh`

2.  **`build_api_lib.sh`** (API Library):
    - Requires `libcausallm_core.so` (run `build_android.sh` first).
    - Compiles **`libcausallm_api.so`** (C-API wrapper).
    - **Usage**: `./build_api_lib.sh`

3.  **`build_test_app.sh`** (Test App):
    - Requires both Core and API libraries.
    - Compiles **`test_api`** (Simple C++ test app for API).
    - **Usage**: `./build_test_app.sh`

4.  **`install_android.sh`**:
    - Installs all built artifacts to a connected Android device.
    - Creates helper scripts (`run_causallm.sh`, `run_test_api.sh`) on the device.
    - **Usage**: `./install_android.sh`

#### Build Instructions

1.  **Set NDK Path**:
    ```bash
    export ANDROID_NDK=/path/to/your/android-ndk
    ```

2.  **Build Core & Main App**:
    ```bash
    cd Applications/CausalLM
    ./build_android.sh
    ```
    Artifacts in `jni/libs/arm64-v8a/`:
    - `libcausallm_core.so`
    - `nntrainer_causallm`

3.  **Build API Library (Optional)**:
    ```bash
    ./build_api_lib.sh
    ```
    Artifacts:
    - `libcausallm_api.so`

4.  **Build Test App (Optional)**:
    ```bash
    ./build_test_app.sh
    ```
    Artifacts:
    - `test_api`

5.  **Install & Run**:
    ```bash
    ./install_android.sh
    ```
    
    **Run Main App:**
    ```bash
    adb shell /data/local/tmp/nntrainer/causallm/run_causallm.sh [model_path]
    ```

    **Run API Test:**
    ```bash
    adb shell /data/local/tmp/nntrainer/causallm/run_test_api.sh [model_name] [prompt]
    ```
## Quantizing Models

NNTrainer provides a quantization utility (`nntr_quantize`) that converts FP32 CausalLM model weights to lower-precision data types, reducing model size for efficient on-device inference.

### Supported Quantization Types

| Data Type | Description |
|-----------|-------------|
| `FP32`    | 32-bit floating point (default for embedding/LM head) |
| `FP16`    | 16-bit floating point |
| `Q4_0`    | 4-bit quantization (default for FC layers) |
| `Q4_K`    | 4-bit K-quant quantization |
| `Q6_K`    | 6-bit K-quant quantization |

> **Note (Q4_0 platform dependency):** `Q4_0` quantization produces platform-specific binary formats — the output generated on x86 is **not compatible** with ARM, and vice versa. You must run `nntr_quantize` on the **same platform architecture** where the quantized model will be used for inference. Cross-platform quantization is not yet supported.


### Prerequisites

The model directory must contain the following files:
- `config.json` – model architecture configuration
- `generation_config.json` – generation parameters
- `nntr_config.json` – NNTrainer-specific configuration
- `.bin` weight file – FP32 model weights

### Building

The quantization utility is built automatically with the CausalLM application:

```bash
meson build && ninja -C build
# The executable is: build/Applications/CausalLM/nntr_quantize
```

### Usage

```
nntr_quantize <model_path> [options]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--output`, `-o <path>` | Output directory | Same as `<model_path>` |
| `--fc_dtype <type>` | Target dtype for FC (fully-connected) layers | `Q4_0` |
| `--embd_dtype <type>` | Target dtype for embedding layer | `FP32` |
| `--lmhead_dtype <type>` | Target dtype for LM head layer | Same as `embd_dtype` |
| `--output_bin <name>` | Output `.bin` filename | Auto-generated |
| `--config <path>` | Use a target `nntr_config.json` for dtype settings | – |

### Examples

```bash
# Quantize FC layers to Q4_0 (default), embedding stays FP32:
nntr_quantize /path/to/qwen3-4b

# Quantize FC layers to Q4_0 and embedding to Q6_K:
nntr_quantize /path/to/qwen3-4b --fc_dtype Q4_0 --embd_dtype Q6_K

# Quantize to a different output directory:
nntr_quantize /path/to/qwen3-4b -o /output/qwen3-4b-q4

# Use a pre-configured target nntr_config.json:
nntr_quantize /path/to/qwen3-4b --config /path/to/target_nntr_config.json
```

### Output

The utility produces:
1. A quantized `.bin` weight file (filename auto-generated or specified via `--output_bin`)
2. A new `nntr_config_quantized.json` (or `nntr_config.json` if output directory differs from source)

After quantization, run the quantized model:
```bash
# If output is in the same directory:
mv /path/to/model/nntr_config_quantized.json /path/to/model/nntr_config.json
nntr_causallm /path/to/model

# If output is in a different directory:
cp /path/to/model/config.json /path/to/model/generation_config.json /output/dir/
nntr_causallm /output/dir
```