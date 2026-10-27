# ☄️ CausalLM Inference with NNTrainer

This application provides a standalone executable and an optional C API to run causal LLM models using NNTrainer.
It supports *inference* mode (text generation) on various devices, including Android.

## Features

- **Standalone Application (`nntr_causallm`)**: A command-line tool to load models and generate text.
- **C API (Optional)**: A lightweight C interface (`libquick_dot_ai_api.so`) for integrating LLM capabilities into other applications (e.g., Android JNI, iOS, or other C/C++ apps).
- **Core Library**: The core implementation is separated into `libcausallm.so` for modularity.
- **Supported Backends**: CPU, **GPU (OpenCL)** — Adreno (Android) and Intel Arc (x86) — and optional QNN on Android. See [README_GPU.md](README_GPU.md) for GPU build/run/env, per-platform performance, and tuning.

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
(32 new tokens). On ARM the prefill attention takes the flash (GEMM) path for
prompts of 32 tokens or more, and the decode step stays on the per-row path.
`"use_flash_attention": false` in `nntr_config.json` keeps the prefill on the
per-row path as well (default `true`).

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

#### Tokenizer library

`Applications/CausalLM/lib/libtokenizers_c.a` is a **tracked** prebuilt of the
in-tree Rust crate `Applications/CausalLM/tokenizers_c_win` (the `_win` suffix is
historical; the crate is the source of truth for every platform). Meson copies it
into `<builddir>/Applications/CausalLM/libtokenizers_c.a` and links the copy, so
no build step names a tracked path as a link input.

The tracked archive is a convenience, so that a checkout builds without a Rust
toolchain. `Applications/CausalLM/tools/build_tokenizers_c.sh` regenerates it
from the crate: `cargo build --locked` pins every dependency to
`tokenizers_c_win/Cargo.lock`, build-machine paths are remapped out of the
archive and debug sections are stripped, so the same rustc and C compiler give
the same bytes on any machine. It also removes the LLVM bitcode sections
(`.llvmbc`/`.llvmcmd`) that the precompiled Rust standard library carries for
LTO and a C/C++ link never reads, about a third of the archive, and fails if the
result is larger than 30 MiB. The script prints the rustc version, the size and
the sha256 of what it built. Requirements: Rust (https://rustup.rs/), plus the NDK
for Android.

```bash
# x86_64 Linux -> Applications/CausalLM/tokenizers-build/x86_64-unknown-linux-gnu/
$ Applications/CausalLM/tools/build_tokenizers_c.sh
# Android arm64 -> Applications/CausalLM/tokenizers-build/aarch64-linux-android/
$ ANDROID_NDK=/path/to/ndk Applications/CausalLM/tools/build_tokenizers_c.sh --abi=arm64-v8a
```

Never write into `Applications/CausalLM/lib/` by hand. A modified archive there
shows up as a dirty 20+ MB tracked file that blocks `git checkout`. To link an
archive you built yourself - for example after changing
`tokenizers_c_win/src/lib.rs`, or for a toolchain the tracked archive does not
match - pass it explicitly:

```bash
$ meson setup build -Denable-transformer=true \
      -Dcausallm-tokenizer-lib=$PWD/Applications/CausalLM/tokenizers-build/x86_64-unknown-linux-gnu/libtokenizers_c.a
```

Refreshing the tracked archive itself is a deliberate, reviewed commit (the
archive has to be rebuilt whenever the crate gains symbols), not a build step:
run the script with `--update-prebuilt` and commit the result together with the
rustc version, size and sha256 it printed.

#### Tokenizer snapshot cache

Parsing a large BPE `tokenizer.json` takes several hundred ms and a few hundred
MB of transient heap on every load. The first load therefore writes a snapshot
of the parsed tables, keyed by the size and mtime of `tokenizer.json`, and later
loads rebuild the tokenizer from it. Anything unexpected in the snapshot (stale
key, truncation, checksum mismatch, a version this build did not write) falls
back to the normal parse and rewrites it; tokenization is identical either way.

- Location: `<tokenizer.json>.ntkc` next to the model, or
  `$XDG_CACHE_HOME/nntrainer/tokenizer/` (else `~/.cache/...`) when the model
  directory is read-only. `NNTR_TOKENIZER_CACHE_DIR=<dir>` makes `<dir>` the only
  location used, for embedders whose model tree is read-only and which have no
  home directory (Android apps, Windows services).
- `NNTR_TOKENIZER_CACHE=0` turns it off at run time.
- Build option `-Dcausallm-tokenizer-cache=auto|enabled|disabled` (default
  `auto`: on when the linked tokenizer library exports the snapshot entry
  points). An archive built before those existed still links, without the cache;
  rebuild it with `tools/build_tokenizers_c.sh`.

### 3. Windows Build & Test

Windows CausalLM builds need a `tokenizers_c.lib` that matches the local
MSVC toolchain. The repository keeps the Linux static library in
`Applications/CausalLM/lib/`; Windows builds generate the matching library from
source instead of carrying a checked-in binary.

The generated library belongs under the build directory (or wherever
`-Dcausallm-tokenizer-lib` points), never in `Applications\CausalLM\lib\`: that
directory holds the tracked Linux archive, and overwriting it leaves a dirty
36+ MB tracked file that blocks `git checkout`.

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

`build_android.sh` is the single entry point for the Android native libraries,
QuickDotAI AAR, and sample app. With no options, it builds the six canonical
CPU native artifacts with Meson, without invoking Gradle or modifying a device.

#### Prerequisites

- Android NDK (`ANDROID_NDK` or `NDK_ROOT`)
- Meson, Ninja, CMake, and Rust (for tokenizers-cpp)
- ADB only when using `--install`
- QNN SDK (`QNN_SDK_ROOT`) only when using `--qnn`

#### Build

```bash
export ANDROID_NDK=/path/to/your/android-ndk
cd Applications/CausalLM
./build_android.sh
```

CPU native artifacts are written under `builddir_app/cpu/`; QNN artifacts use
`builddir_app/qnn/`. To additionally build the standalone app and AAR without
installing them, run `./build_android.sh --app`. That mode stages
`libcausallm.so`, `libquick_dot_ai_api.so`, and their runtime dependencies;
Gradle adds `libquickai_jni.so` and writes the debug AAR under
`Android/QuickDotAI/build/outputs/aar/`.

The most useful options are:

| Option | Behavior |
|---|---|
| `--app` | Add the QuickDotAI AAR and sample APK to the canonical native build. |
| `--install` | Push the native libraries and tools; with `--app`, also install the built APK. |
| `--qnn` | Build the QNN variant and include its runtime libraries; requires `QNN_SDK_ROOT`. CPU is the default. |
| `--cache` | Reuse a compatible existing nntrainer Android engine build, or build it when absent. |
| `--clean` | Recreate the selected CPU or QNN CausalLM build directory. |
| `--nntr-threads=N` | Set the positive nntrainer compute-thread count. |

`--app`, `--install`, and QNN selection are independent. For example,
`--install` pushes only native artifacts, while `--app --install` also installs
the APK. Set `ANDROID_SERIAL` when more than one device is connected.
QNN deployment supports HTP V75, V79, and V81 SDK runtimes and requires at
least one complete matching Stub/Skel pair.

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
| `Q8_0`    | 8-bit block quantization |

> **Note (Q4_0 / Q8_0 platform dependency):** `Q4_0` and `Q8_0` quantization produce platform-specific binary layouts — FC weights are repacked for the target ISA (`Q4_0`: `q4_0x8` on x86 / `q4_0x4` on ARM; `Q8_0`: plain rows on x86 / `q8_0x4` interleaved on ARM). A file produced for one platform is **not compatible** with the other and silently yields garbage output (the bytes parse, the numbers are wrong). Either run `nntr_quantize` on the same platform architecture where the model will be used, or pass `--isa ARM` / `--isa X86` explicitly when cross-quantizing (with no `--isa`, `DEFAULT` resolves to the **build host's** layout — quantizing on an x86 host for an Android device requires `--isa ARM`).


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
| `--no_sidecar` | Keep the embedding and per-layer-embedding tables inside the `.bin` (see below) | sidecar when eligible |
| `--no_ple_sidecar` / `--no_embd_sidecar` | Keep only that one table inside the `.bin` | sidecar when eligible |
| `--ple_sidecar` / `--embd_sidecar` | Require that table in a sidecar; fail if the model is not eligible | – |

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

### Embedding sidecar (default)

When the embedding is quantized to `Q4_0` or `Q6_K` and the output is a `.bin`,
`nntr_quantize` writes two tables to their own files next to the model file
instead of into it:

- the per-layer-embedding (PLE) table of a model that has one
  (`hidden_size_per_layer_input` in `config.json`, e.g. Gemma-4):
  `<bin>_ple.bin` + `<bin>_ple.json`, named by `ple_file_name`;
- the `embedding0` token-embedding table of a model with an untied lm_head
  (`lmhead_untie: true`): `<bin>_embd.bin` + `<bin>_embd.json`, named by
  `embedding_file_name`.

The runtime mmaps these files and reads only the rows of the tokens it sees, so
no backend keeps a resident copy of the table. The tokens are identical to the
in-file layout. For Gemma-4 (QS4CX FC / lm_head, Q6_K embedding) this moves
about 2.1 GiB out of the model file and, on the same binary and 1K prompt,
lowers peak host or device memory by 40-64 % and model init by 36-76 % on the
x86 and Arm CPU, OpenCL and CUDA backends. A tied lm_head reads every embedding row at each decode step, so
`embedding0` stays in the model file for tied models; models without the
qualifying tables (Qwen3, Gemma-3, ...) are written exactly as before. Pass
`--no_sidecar` to keep the former single-file layout. Both layouts load: the
runtime follows `ple_file_name` / `embedding_file_name` in `nntr_config.json`,
resolved against the model directory, and reads the tables from the `.bin`
when the keys are absent.

### Output

The utility produces:
1. A quantized `.bin` weight file (filename auto-generated or specified via `--output_bin`)
2. A new `nntr_config_quantized.json` (or `nntr_config.json` if output directory differs from source)
3. For an eligible model, the embedding sidecar files described above

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

`Q4_0` (x86: `q4_0x8`, ARM: `q4_0x4`) and `Q8_0` FC weights (x86: plain rows,
ARM: `q8_0x4` interleaved) are repacked into ISA-specific layouts that the
header bytes alone cannot distinguish, so files containing such tensors record
`nntr_q4_0_isa` / `nntr_q8_0_isa` (`x86` / `arm`) under `__metadata__`. This is
the layout chosen by `--isa` (with `DEFAULT` resolving to the build platform),
so a file cross-quantized on x86 with `--isa ARM` is tagged `arm` and is
identifiable before it is loaded on the wrong architecture. (`Q8_0` embedding
lookup tables always stay in the plain row layout; the tag describes the
FC/GEMM weights.)

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
