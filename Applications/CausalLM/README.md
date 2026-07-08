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

## Performance

Measured on a **Galaxy S26 Ultra (SM-S948U)**, CPU backend, with **Qwen3-0.6B**
quantized to **Q4_0** FC weights + **Q6_K** embedding / LM head. `prefill` =
prompt-encode throughput, `decode` = autoregressive generation throughput
(32 new tokens). Flash (GEMM) attention engages for prompts ≥ 32 tokens.

| Activation | Threads | Prompt | Prefill (tok/s) | Decode (tok/s) |
| --- | --- | --- | --- | --- |
| **Q4_0-FP16** | 8 | 437 | **755** | **80** |
| Q4_0-FP16 | 8 | 1003 | 596 | 60 |
| Q4_0-FP16 | 4 | 1003 | 423 | 52 |
| Q4_0-FP32 | 8 | 437 | 329 | 77 |
| Q4_0-FP32 | 8 | 1003 | 299 | 50 |
| Q4_0-FP32 | 4 | 1003 | 206 | 45 |

`Q4_0-FP16` (FP16 activation) is the recommended device config: ~2× the prefill
throughput of the FP32-activation path on the FP16 build, and token-coherent with
it. Prefill throughput drops as the prompt grows (attention is O(n²)); a very
short prompt (e.g. the 18-token default `sample_input`) reports a much lower
number (~240 tok/s) because it is below the flash-attention threshold and is
dominated by fixed setup cost — not representative of sustained throughput.
Peak RSS ≈ 0.94 GB.

## CausalLM API

The CausalLM application exposes a C API for easy integration with other applications (e.g., Android JNI).
The API allows loading models, running inference, and retrieving performance metrics.

For detailed documentation, please refer to [API Documentation](api/README.md).

## Chat Template

CausalLM supports automatic chat template formatting by reading the `chat_template` field from HuggingFace's `tokenizer_config.json`. This eliminates the need for hardcoded per-model chat formatting.

### How It Works

Most HuggingFace models include a `tokenizer_config.json` with a `chat_template` field (Jinja2 format) that defines how to format conversations. CausalLM includes a built-in mini Jinja2 renderer that processes these templates at runtime.

When a `tokenizer_config.json` is present in the model directory:
- **CLI (`nntr_causallm`)**: Raw user input provided as a command-line argument is automatically wrapped with the chat template.
- **C API**: The `apply_chat_template()` function uses the dynamic template instead of hardcoded formats.

If `tokenizer_config.json` is absent or does not contain a `chat_template` field, a warning is printed and raw input is passed through unchanged.

### Supported Template Features

The built-in Jinja2 renderer supports the following constructs commonly used in HuggingFace chat templates:

| Feature | Example |
|---------|---------|
| For loops | `{% for message in messages %}...{% endfor %}` |
| Conditionals | `{% if %}...{% elif %}...{% else %}...{% endif %}` |
| Output expressions | `{{ bos_token }}` |
| Variable assignment | `{% set offset = 1 %}` |
| Dict/array access | `message['role']`, `messages[0]` |
| String concatenation | `'<\|im_start\|>' + message['role']` |
| Comparison operators | `==`, `!=`, `>`, `<`, `>=`, `<=` |
| Boolean operators | `and`, `or`, `not` |
| Loop variables | `loop.first`, `loop.last`, `loop.index`, `loop.index0` |
| Filters | `\| trim`, `\| length`, `\| tojson` |
| String methods | `.strip()`, `.startswith()`, `.upper()`, `.split()` |
| Containment test | `'keyword' in message['content']` |
| Namespace | `namespace()` for cross-scope variable mutation |
| Whitespace control | `{%- -%}`, `{{- -}}` |

### Required Files

To use chat templates, ensure `tokenizer_config.json` is in your model directory alongside the other config files. This file is included by default when downloading models from HuggingFace.

### Example

```bash
# With tokenizer_config.json present, raw input is auto-formatted:
./nntr_causallm /path/to/model "What is machine learning?"

# The input will be automatically wrapped, e.g. for Qwen3:
# <|im_start|>user
# What is machine learning?<|im_end|>
# <|im_start|>assistant
```

### Multi-turn Conversations (API)

The C API supports multi-turn conversations through `ChatMessage`:

```cpp
#include "chat_template.h"

causallm::ChatTemplate tmpl = causallm::ChatTemplate::fromFile("tokenizer_config.json");

std::vector<causallm::ChatMessage> messages = {
  {"system", "You are a helpful assistant."},
  {"user", "Hello!"},
  {"assistant", "Hi there!"},
  {"user", "How are you?"}
};

std::string formatted = tmpl.apply(messages);
```

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

### 3. Windows Build & Test

Windows CausalLM builds need a `tokenizers_c.lib` that matches the local
MSVC toolchain. The repository keeps the Linux static library in
`Applications/CausalLM/lib/`; Windows builds generate the matching library from
source instead of carrying a checked-in binary.

#### Prerequisites

- Visual Studio Build Tools with the MSVC C++ toolchain
- Meson and Ninja
- Rust (`cargo`) from https://rustup.rs/

#### Build tokenizer library

Meson builds the default `tokenizers_c.lib` automatically when it is missing.
The helper can also be run directly to pre-build or refresh the library:

```powershell
PS> powershell -ExecutionPolicy Bypass -File Applications\CausalLM\build_tokenizer_windows.ps1 -BuildDir build-causallm-win
```

The build writes the default Meson input under the build directory:

```text
build-causallm-win\tokenizers_c_win\target\release\tokenizers_c.lib
```

For Windows cross builds, Meson passes the matching Rust target triple and the
library is written under `target\<triple>\release\`.

If you already have a compatible `tokenizers_c.lib`, pass it explicitly during
Meson setup:

```powershell
PS> meson setup build-causallm-win -Dplatform=windows -Denable-transformer=true -Dcausallm-tokenizer-lib=C:\path\to\tokenizers_c.lib
```

When using a DLL import library instead of a static library, make sure the
matching `tokenizers_c.dll` is available on `PATH` at runtime.

#### Build and run

```powershell
PS> meson setup build-causallm-win -Dplatform=windows -Denable-transformer=true -Denable-test=false
PS> ninja -C build-causallm-win nntr_causallm
PS> $build = Resolve-Path build-causallm-win
PS> $dllDirs = Get-ChildItem $build -Filter *.dll -Recurse | ForEach-Object { Split-Path -Parent $_.FullName } | Sort-Object -Unique
PS> $env:PATH = (($dllDirs + @($build, "$build\Applications\CausalLM")) -join ";") + ";" + $env:PATH
PS> $env:NNTR_NUM_THREADS = "4"
PS> .\build-causallm-win\Applications\CausalLM\nntr_causallm.exe C:\path\to\model "Hello from Windows"
```

### 4. Android Build & Test

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
| `--output_bin <name>` | Output weight filename | Auto-generated |
| `--output_format <bin\|safetensors>` | Output weight container format | `bin` |
| `--config <path>` | Use a target `nntr_config.json` for dtype settings | – |
| `--isa <x86|ARM|DEFAULT>` | Target ISA for quantization | `DEFAULT` |

> The input weight format (`.bin` or `.safetensors`) is auto-detected from the
> file referenced by `model_file_name` in `nntr_config.json`, so any of the four
> input/output combinations is supported.

### Examples
```bash
# Quantize FC layers to Q4_0 (default), embedding stays FP32:
nntr_quantize /path/to/qwen3-4b

# Quantize FC layers to Q4_0 and embedding to Q6_K:
nntr_quantize /path/to/qwen3-4b --fc_dtype Q4_0 --embd_dtype Q6_K

# Quantize FC layers to Q4_0 and embedding to Q6_K in ARM format:
nntr_quantize /path/to/qwen3-4b --fc_dtype Q4_0 --embd_dtype Q6_K --isa ARM

# Quantize to a different output directory:
nntr_quantize /path/to/qwen3-4b -o /output/qwen3-4b-q4

# Use a pre-configured target nntr_config.json:
nntr_quantize /path/to/qwen3-4b --config /path/to/target_nntr_config.json

# Quantize FC layers to Q4_0 and write a .safetensors file instead of .bin:
nntr_quantize /path/to/qwen3-4b --fc_dtype Q4_0 --output_format safetensors
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

# If output is in a different directory (-o), the tool copies config.json,
# generation_config.json and the tokenizer files automatically, so the
# output directory is self-contained:
nntr_causallm /output/dir
```

## Quantized Safetensors Format

NNTrainer can store quantized weights (`Q4_0` / `Q4_K` / `Q6_K`) in the
[safetensors](https://github.com/huggingface/safetensors) container in addition
to the raw `.bin` format. The quantized payload is byte-for-byte identical to
the `.bin` payload — only the container differs.

### How it fits together

```
                      ┌──────────────────────────────┐
   FP32 weights ─────▶│          nntr_quantize         │
 (.bin / .safetensors)│  GgmlQuantizer (Q4_0/Q4_K/Q6_K)│
                      └───────────────┬────────────────┘
                                      │  --output_format
                          ┌───────────┴───────────┐
                          ▼                         ▼
                  quantized .bin           quantized .safetensors
                                            (self-describing header)
                                                    │
                  ┌─────────────────────────────────┤
                  ▼                                  ▼
       nntr_safetensors_info               nntr_causallm (runtime)
       (header-only inspection)        1. read header  → byte offsets
                                       2. mmap data section (no full read)
                                       3. weight tensors point at the
                                          quantized blocks directly
```

At load time the runtime only parses the (small) JSON header to obtain each
tensor's byte offset, then memory-maps the data section — the large file is
**never read twice**.

### Header layout

A safetensors file is `[8-byte header length][JSON header][packed tensor data]`.
Quantized tensors are stored as opaque byte blobs so that standard safetensors
tooling can still read the file, while the native nntrainer type and logical
(pre-quantization) shape are preserved in extension fields:

```json
{
  "__metadata__": {
    "format": "nntrainer",
    "nntr_format": "nntr-safetensors-v1",
    "nntr_q4_0_isa": "arm"
  },
  "layer0_wq:weight": {
    "dtype": "U8",
    "shape": [2359296],
    "nntr_dtype": "Q4_0",
    "nntr_shape": [1, 1, 1024, 4096],
    "data_offsets": [0, 2359296]
  },
  "output_norm:weight": {
    "dtype": "F32",
    "shape": [1, 1, 1, 1024],
    "data_offsets": [2359296, 2363392]
  }
}
```

| Field | Meaning |
|-------|---------|
| `dtype` | Standard safetensors dtype. `U8` for any block-quantized tensor. |
| `shape` | Standard shape. For quantized tensors this is the raw byte length. |
| `nntr_dtype` | Native nntrainer type (`Q4_0` / `Q4_K` / `Q6_K`). Absent for FP32/FP16. |
| `nntr_shape` | Logical (pre-quantization) `[N, C, H, W]` shape. Absent for FP32/FP16. |
| `data_offsets` | `[start, end)` byte range within the data section. |

FP32/FP16 tensors are written with their standard `dtype`/`shape` and no
extension fields, so plain (non-quantized) files stay fully standard.

`Q4_0` is repacked into an ISA-specific layout (x86: `q4_0x8`, ARM: `q4_0x4`)
that the header bytes alone cannot distinguish, so files containing a `Q4_0`
tensor record `nntr_q4_0_isa` (`x86` / `arm`) under `__metadata__`. This is the
layout chosen by `--isa` (with `DEFAULT` resolving to the build platform), so a
file cross-quantized on x86 with `--isa ARM` is tagged `arm` and is identifiable
before it is loaded on the wrong architecture.

### Inspecting a file

Use `nntr_safetensors_info` to read just the header and print the embedded
metadata plus a per-tensor table — no weight data is loaded:

```bash
nntr_safetensors_info /path/to/model.safetensors
```

```
file: model.safetensors
header bytes: 24960

metadata:
  format = nntrainer
  nntr_format = nntr-safetensors-v1
  nntr_q4_0_isa = arm

tensors: 2
  name                 dtype     bytes         shape
  layer0_wq:weight     Q4_0      2359296       [1,1,1024,4096]
  output_norm:weight   F32       4096          [1,1,1,1024]
```

This makes a quantized `.safetensors` file self-describing: the quantization
type of each weight is visible without an accompanying `nntr_config.json`.
